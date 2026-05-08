#!/usr/bin/env python3
# This file is part of the ACTS project.
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from pathlib import Path

os.environ["ACTS_SEQUENCER_DISABLE_FPEMON"] = "1"

import acts
import acts.examples
from acts import UnitConstants as u

import numpy as np
import torch

import matplotlib.pyplot as plt

from ml_utilities import (
    DataHandler,
)


def runTrackFindingPythonOnly(
    trackingGeometry,
    field,
    digiConfigFile,
    geoSelectionConfigFile,
    outputDir,
    decorators=[],
    s=None,
):
    from acts.examples.simulation import (
        addParticleGun,
        MomentumConfig,
        EtaConfig,
        PhiConfig,
        ParticleConfig,
        addFatras,
        addDigitization,
    )

    from regressor_models import (
        MLP,
        printModelSummary
    )

    s = s or acts.examples.Sequencer(events=1, numThreads=1, logLevel=acts.logging.INFO)
    outputDir = Path(outputDir)
    rnd = acts.examples.RandomNumbers(seed=42)

    for d in decorators:
        s.addContextDecorator(d)

    addParticleGun(
        s,
        MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
        EtaConfig(-2.0, 2.0, uniform=True),
        PhiConfig(0.0, 360.0 * u.degree),
        ParticleConfig(1, acts.PdgParticle.eMuon, randomizeCharge=True),
        rnd=rnd,
    )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
    )

    s.addAlgorithm(
        acts.examples.SpacePointMaker(
            level=acts.logging.INFO,
            trackingGeometry=trackingGeometry,
            inputMeasurements="measurement_subset",
            outputSpacePoints="spacepoints",
            geometrySelection=acts.examples.json.readJsonGeometryList(
                str(geoSelectionConfigFile)
            ),
        )
    )

    # class PythonTrackFinder(acts.examples.IAlgorithm):
    #     def __init__(self, name, level):
    #         acts.examples.IAlgorithm.__init__(self, name, level)

    #         self.spacepoints = acts.examples.ReadDataHandle(
    #             self, acts.SpacePointContainer2, "Spacepoints"
    #         )
    #         self.spacepoints.initialize("spacepoints")

    #         self.prototracks = acts.examples.WriteDataHandle(
    #             self, acts.examples.ProtoTrackContainer, "Prototracks"
    #         )
    #         self.prototracks.initialize("prototracks")

    #     def execute(self, context):
    #         spacepoints = self.spacepoints(context.eventStore)

    #         track = acts.examples.ProtoTrack()
    #         for sp in sorted(spacepoints, key=lambda sp: sp.r):
    #             for sl in sp.sourceLinks:
    #                 isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
    #                 track.append(isl.index())

    #         prototracks = acts.examples.ProtoTrackContainer()
    #         prototracks.append(track)

    #         self.prototracks(context, prototracks)
    #         return acts.examples.ProcessCode.SUCCESS

    # s.addAlgorithm(PythonTrackFinder("PythonTrackFinder", acts.logging.INFO))

    trkParamExtractor = acts.examples.ParticleTrackParamExtractor(
        level=acts.logging.INFO,
        inputParticles="particles_generated_selected",
        outputTrackParameters="true_parameters",
    )
    s.addAlgorithm(trkParamExtractor)

    truthTrkFndAlg = acts.examples.TruthTrackFinder(
        level=acts.logging.INFO,
        inputParticles="particles_generated_selected",
        inputMeasurements="measurements",
        inputParticleMeasurementsMap="particle_measurements_map",
        inputSimHits="simhits",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        # outputProtoTracks="prototracks",
        outputProtoTracks="truth_particle_tracks",
    )
    s.addAlgorithm(truthTrkFndAlg)

    class PythonTrackFitter(acts.examples.IAlgorithm):
        def __init__(self, name, level):
            acts.examples.IAlgorithm.__init__(self, name, level)

            self.prototracks = acts.examples.ReadDataHandle(
                self, acts.examples.ProtoTrackContainer, "Prototracks"
            )
            # self.prototracks.initialize("prototracks")
            self.prototracks.initialize("truth_particle_tracks")

            self.tracks = acts.examples.WriteDataHandle(
                self, acts.examples.ConstTrackContainer, "Tracks"
            )
            self.tracks.initialize("fitted_tracks")

            # NEW
            self.spacepoints = acts.examples.ReadDataHandle(
                self, acts.SpacePointContainer2, "Spacepoints"
            )
            self.spacepoints.initialize("spacepoints")

            self.max_seq_len = 20
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.mlp = MLP(input_dim=max_seq_len*3, output_dim=5, hidden_dim=32, n_hidden_layers=2)
            self.mlp = MLP(input_dim=self.max_seq_len*3, output_dim=5, hidden_dim=256, n_hidden_layers=7)
            self.mlp.to(device)
            self.mlp.load_state_dict(torch.load(mlModelFile, map_location=device))

        def execute(self, context):
            prototracks = self.prototracks(context.eventStore)
            spacepoints = self.spacepoints(context.eventStore)

            # Build a mapping from measurement index to the corresponding
            # space point produced by SpacePointMaker. Each space point carries
            # one or more SourceLinks to the original measurement indices.
            measurement_to_spacepoint = {}
            measurement_to_sourcelink = {}
            for sp in spacepoints:
                for sl in sp.sourceLinks:
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    meas_id = isl.index()
                    measurement_to_spacepoint[meas_id] = sp
                    measurement_to_sourcelink[meas_id] = sl

            # LOOK AT THE CODE FOR THIS ONE
            container = acts.examples.TrackContainer()
            print(prototracks)
            surface_map = trackingGeometry.geoIdSurfaceMap()
            print(surface_map)

            tech_acts_dir = "/home/taleiko/Documents/CERN/Technical_Student/Program/acts"
            train_data_dirs = [os.path.join(tech_acts_dir, "mega_data/mega_data_{}/{}/{}/train_100000".format(str(num), "electron", "geant4")) for num in range(10)]

            dh = DataHandler(
                train_data_dirs,
                load_data_scalers=True
            )
            input_scaler = dh.getInputScaler()
            output_scaler = dh.getOutputScaler()

            for prototrack in prototracks:
                ml_input = np.array([[
                    measurement_to_spacepoint[meas_id].x,
                    measurement_to_spacepoint[meas_id].y,
                    measurement_to_spacepoint[meas_id].z,
                ] for meas_id in prototrack])

                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(111, projection='3d')
                for coord in ml_input:
                    ax.scatter(coord[0], coord[1], coord[2])
                # plt.show()
                plt.savefig("/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code/trajectory.png")

                # print(ml_input)

                ml_input = np.flip(ml_input, axis=0)
                # print(ml_input)
                scaled_input = input_scaler.transform(ml_input)
                # print(scaled_input)
                pad_len = self.max_seq_len - len(scaled_input)
                scaled_input = np.pad(scaled_input, ((0, pad_len), (0, 0)), mode='constant')
                scaled_input = scaled_input.flatten()
                # print(scaled_input)
                scaled_input = torch.tensor(scaled_input, dtype=torch.float32)
                # print(scaled_input)

                scaled_output = self.mlp(scaled_input)
                scaled_output.to(torch.device("cpu"))
                o = scaled_output.detach().numpy()
                # print(o)
                o = np.array([o])
                # print(o)
                output = output_scaler.inverse_transform(o)
                # print(output)
                output = output[0]
                # print(output)

                track = container.makeTrack()
                track.parameters = acts.BoundVector(output[0], output[1], output[2], output[3], output[4], 1.0)
                track.nMeasurements = len(prototrack)

                # Attach measurements to the track state. Use the original source
                # link from the reconstructed space point and the matching
                # surface from the geometry map.
                for meas_id in prototrack:
                    sp = measurement_to_spacepoint[meas_id]
                    sl = measurement_to_sourcelink[meas_id]
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    sf = surface_map[isl.geometryId()]

                    trackState = track.appendTrackState()
                    trackState.setIsMeasurement()
                    trackState.setUncalibratedSourceLink(sl)
                    trackState.setReferenceSurface(sf)

            self.tracks(context, container.makeConst())
            return acts.examples.ProcessCode.SUCCESS

    s.addAlgorithm(PythonTrackFitter("PythonTrackFitter", acts.logging.INFO))

    # SET TO VERBOSE (DEBUG) MODE
    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            # level=acts.logging.INFO,
            level=acts.logging.VERBOSE,
            inputTracks="fitted_tracks",
            inputParticles="particles",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="track_particle_matching",
            outputParticleTrackMatching="particle_track_matching",
            doubleMatching=True,
        )
    )

    # SET TO VERBOSE (DEBUG) MODE
    cfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    cfg.inputTracks = "fitted_tracks"
    cfg.inputParticles = "particles"
    cfg.inputTrackParticleMatching = "track_particle_matching"
    cfg.inputParticleTrackMatching = "particle_track_matching"
    cfg.inputParticleMeasurementsMap = "particle_measurements_map"
    perfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        # cfg, acts.logging.INFO
        cfg, acts.logging.VERBOSE
    )
    s.addWriter(perfWriter)

    return s, perfWriter


if __name__ == "__main__":
    # srcdir = Path(__file__).resolve().parent.parent.parent.parent
    # srcdir = Path(__file__).resolve().parent.parent / "Technical_student" / "Program" / "acts"
    # srcdir = Path(__file__).resolve() / "acts"
    # srcdir = Path(__file__).resolve().parent
    srcdir = Path("/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code")
    # print(srcdir)
    # sys.exit(0)

    detector = acts.examples.GenericDetector(acts.examples.GenericDetector.Config())
    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()

    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))



    digiConfigFile = srcdir / "generic-digi-smearing-config.json"
    geoSelectionConfigFile = srcdir / "generic-pixel-sstrips-lstrips-spacepoints.json"
    mlModelFile = "/home/taleiko/Documents/CERN/Technical_Student/Resultat/mega_mlp_1000e_8h_256n_0.001lr_1024b/mega_mlp_1000e_8h_256n_0.001lr_1024b.pt"

    outputDir = Path.cwd() / "output_track_finding_python_only"
    outputDir.mkdir(exist_ok=True)

    s, perfWriter = runTrackFindingPythonOnly(
        trackingGeometry=trackingGeometry,
        field=field,
        digiConfigFile=digiConfigFile,
        geoSelectionConfigFile=geoSelectionConfigFile,
        outputDir=outputDir,
        decorators=decorators,
    )
    s.run()

    print(perfWriter.histograms().keys())
    fig, ax = plt.subplots()
    # print(type(histWriter.histograms()['trackeff_vs_eta'].plot(ax=ax)))
    perfWriter.histograms()['trackeff_vs_eta'].plot(ax=ax)
    # sys.exit(0)
    # histWriter.histograms()['trackeff_vs_eta'].plot(ax=ax)
    # plt.show()
    # ax.set_xlim(-0.1, 0.1)
    plt.savefig("/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code/ml_hist.png")
    


    # histograms = perfWriter.histograms()
    # print(
    #     f"Retrieved {len(histograms)} performance histograms: {list(histograms.keys())}"
    # )
    # h = histograms["trackeff_vs_DeltaR"]
    # print(dir(h))
    # # print(h.accepted)
    # # print(h.name)
    # # print(h.plot)
    # # print(h.rank)
    # # print(h.title)
    # print(h.total)
    # print(h.total)

    # print(h.plot)
    # print(h.plot())

    # num = h.accepted
    # den = h.total

    # print(dir(num))
    # print(num.values)
    # print(num.values())
