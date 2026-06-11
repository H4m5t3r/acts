#!/usr/bin/env python3
# This file is part of the ACTS project.
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
from pathlib import Path
from typing import Optional
import argparse
import warnings

os.environ["ACTS_SEQUENCER_DISABLE_FPEMON"] = "1"

import acts
import acts.examples
from acts import UnitConstants as u
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.root import (
    RootParticleReader,
    RootSimHitReader,
)

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ml_utilities import (
    DataHandler,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--odd",
    help="Switch use of the ODD on/off",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--material-config", help="Material map configuration file", type=Path
)
parser.add_argument(
    "--output-root",
    help="Switch root output on/off",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--output-csv",
    help="Switch csv output on/off",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--read-data",
    help="Switch data reading instead of simulation on/off",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--n-events",
    help="Set the number of events to be simulated if data is not read",
    default=1,
    type=int,
)
parser.add_argument(
    "--random-seed",
    help="Set the random seed for data generation",
    default=42,
    type=int,
)
parser.add_argument(
    "--mode",
    help="Set the simulation mode",
    default="geant4",
    type=str,
)
args = parser.parse_args()


def runTrackFindingPythonOnly(
    trackingGeometry,
    field,
    digiConfigFile,
    geoSelectionConfigFile,
    outputDir,
    mlModelFile,
    inputParticlePath: Optional[Path] = None,
    inputSimHitsPath: Optional[Path] = None,
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

    from acts.examples.root import (
        RootParticleReader,
        # RootTrackStatesWriter,
        # RootTrackSummaryWriter,
        # RootTrackFitterPerformanceWriter,
    )

    from regressor_models import MLP, printModelSummary

    if args.read_data:
        # Previous data scalers loaded here, will have to initialize with new directory to create new scalers
        dataHandler = DataHandler([dataDir], load_data_scalers=True)
        mlInputs = dataHandler.readX([dataDir])
        n_events = len(mlInputs)
    else:
        n_events = 1

    s = s or acts.examples.Sequencer(
        events=n_events, numThreads=1, logLevel=acts.logging.INFO
    )
    # mode = "geant4"
    # s = s or acts.examples.Sequencer(
    #     events=n_events,
    #     numThreads=1 if mode == "geant4" else -1,
    #     logLevel=acts.logging.INFO,
    # )
    outputDir = Path(outputDir)
    rnd = acts.examples.RandomNumbers(seed=42)
    logger = acts.getDefaultLogger("Python Tracking Example", acts.logging.INFO)

    for d in decorators:
        s.addContextDecorator(d)

    if args.read_data:
        inputParticlePath = outputDir / "root" / "particles.root"
        inputSimHitsPath = outputDir / "hits.root"

    if inputParticlePath is None:
        logger.info("Generating particles with addParticleGun()")
        addParticleGun(
            s,
            MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
            EtaConfig(-2.0, 2.0, uniform=True),
            PhiConfig(0.0, 360.0 * u.degree),
            ParticleConfig(1, acts.PdgParticle.eMuon, randomizeCharge=True),
            rnd=rnd,
            # ADDED
            outputDirCsv=outputDir / "csv",
            outputDirRoot=outputDir / "root",
        )
    else:
        logger.info("Reading particles from {}", inputParticlePath.resolve())
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                outputParticles="particles_generated_selected",
            )
        )
        s.addWhiteboardAlias("particles", "particles_generated_selected")

    if inputSimHitsPath is None:
        addFatras(
            s,
            trackingGeometry,
            field,
            rnd=rnd,
            # ADDED
            outputDirRoot=outputDir,
        )
    else:
        logger.info("Reading hits from {}", inputSimHitsPath.resolve())
        s.addReader(
            RootSimHitReader(
                level=acts.logging.INFO,
                filePath=str(inputSimHitsPath.resolve()),
                outputSimHits="simhits",
            )
        )
        s.addWhiteboardAlias("particles_simulated_selected", "particles_generated")

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
        # ADDED
        outputDirCsv=outputDir / "csv",
        outputDirRoot=outputDir / "root",
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

    # NOT NEEDED?
    # Codex: It would only be needed if you later add an algorithm that consumes generated-particle track parameters, for example truth-smearing / seed-parameter estimation, propagation from truth parameters, material validation, or a writer/printer for those parameters
    # trkParamExtractor = acts.examples.ParticleTrackParamExtractor(
    #     level=acts.logging.INFO,
    #     inputParticles="particles_generated_selected",
    #     outputTrackParameters="true_parameters",
    # )
    # s.addAlgorithm(trkParamExtractor)

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
            self.mlp = MLP(
                input_dim=self.max_seq_len * 3,
                output_dim=5,
                hidden_dim=256,
                n_hidden_layers=7,
            )
            self.mlp.to(device)
            self.mlp.load_state_dict(torch.load(mlModelFile, map_location=device))

        def execute(self, context):
            prototracks = self.prototracks(context.eventStore)
            spacepoints = self.spacepoints(context.eventStore)

            measurement_to_spacepoint = {}
            measurement_to_sourcelink = {}
            for sp in spacepoints:
                for sl in sp.sourceLinks:
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    meas_id = isl.index()
                    measurement_to_spacepoint[meas_id] = sp
                    measurement_to_sourcelink[meas_id] = sl

            container = acts.examples.TrackContainer()
            surface_map = trackingGeometry.geoIdSurfaceMap()
            print(prototracks)

            tech_acts_dir = (
                "/home/taleiko/Documents/CERN/Technical_Student/Program/acts"
            )
            train_data_dirs = [
                os.path.join(
                    tech_acts_dir,
                    "mega_data/mega_data_{}/{}/{}/train_100000".format(
                        str(num), "electron", "geant4"
                    ),
                )
                for num in range(10)
            ]

            # Glued together to work: Data directories not actually used here
            # NOTE: Now potential double data reading if new scalers are actually created
            # See earlier DataHandler
            dh = DataHandler(train_data_dirs, load_data_scalers=True)
            input_scaler = dh.getInputScaler()
            output_scaler = dh.getOutputScaler()

            print(measurement_to_spacepoint)
            for prototrack in prototracks:
                ml_input = np.array(
                    [
                        [
                            measurement_to_spacepoint[meas_id].x,
                            measurement_to_spacepoint[meas_id].y,
                            measurement_to_spacepoint[meas_id].z,
                        ]
                        for meas_id in prototrack
                    ]
                )
                print([meas_id for meas_id in prototrack])
                print(ml_input)
                # sys.exit(0)

                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111, projection="3d")
                for coord in ml_input:
                    ax.scatter(coord[0], coord[1], coord[2])
                # plt.show()
                plt.savefig(
                    "/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code/trajectory.png"
                )

                # print(ml_input)

                ml_input = np.flip(ml_input, axis=0)
                # print(ml_input)
                scaled_input = input_scaler.transform(ml_input)
                # print(scaled_input)
                pad_len = self.max_seq_len - len(scaled_input)
                scaled_input = np.pad(
                    scaled_input, ((0, pad_len), (0, 0)), mode="constant"
                )
                scaled_input = scaled_input.flatten()
                # print(scaled_input)
                scaled_input = torch.tensor(scaled_input, dtype=torch.float32)
                # print(scaled_input)

                # TODO: This is taped together at the moment. The scaler expects an array of columns. Note output[0] and array([output])
                with torch.no_grad():
                    scaled_output = np.array(
                        [self.mlp(scaled_input).detach().cpu().numpy()]
                    )
                output = output_scaler.inverse_transform(scaled_output)
                # print(output)
                output = output[0]
                # print(output)

                track = container.makeTrack()
                track.parameters = acts.BoundVector(
                    output[0], output[1], output[2], output[3], output[4], 1.0
                )
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
                    trackState.typeFlags.isMeasurement = True
                    trackState.uncalibratedSourceLink = sl
                    trackState.referenceSurface = sf

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
        cfg,
        acts.logging.VERBOSE,
    )
    s.addWriter(perfWriter)

    return s, perfWriter


def runOddGsfTrackFinding(
    trackingGeometry,
    field,
    digiConfigFile,
    geoSelectionConfigFile,
    stripGeoSelectionConfigFile,
    outputDir,
    mlModelFile,
    inputParticlePath: Optional[Path] = None,
    inputSimHitsPath: Optional[Path] = None,
    decorators=[],
    s=None,
):
    from acts.examples.simulation import (
        addParticleGun,
        ParticleConfig,
        EtaConfig,
        PhiConfig,
        MomentumConfig,
        addFatras,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
        addGeant4,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TrackSmearingSigmas,
        addTruthTrackingGsf,
    )
    from acts.examples.root import (
        RootTrackStatesWriter,
        RootTrackSummaryWriter,
        RootTrackFitterPerformanceWriter,
    )

    from regressor_models import MLP, printModelSummary

    if args.read_data:
        # Previous data scalers loaded here, will have to initialize with new directory to create new scalers
        dataHandler = DataHandler([dataDir], load_data_scalers=True)
        mlInputs = dataHandler.readX([dataDir])
        n_events = len(mlInputs)
    else:
        n_events = args.n_events

    # mode = "fatras"
    # mode = "geant4"
    mode = args.mode
    s = s or acts.examples.Sequencer(
        events=n_events,
        numThreads=1 if mode == "geant4" else -1,
        logLevel=acts.logging.INFO,
    )

    for d in decorators:
        s.addContextDecorator(d)

    outputDir = outputDir / "electron"
    data_type = "train"
    if mode == "fatras":
        outputDir = outputDir / "fatras"
    else:
        outputDir = outputDir / "geant4"
    print(outputDir)
    print(type(outputDir))
    if n_events == 10000:
        dir_ending = ""
    else:
        dir_ending = "_{}".format(str(n_events))
    if data_type == "train":
        outputDir = outputDir / ("train" + dir_ending)
    else:
        outputDir = outputDir / ("test" + dir_ending)
    print(outputDir)
    print(type(outputDir))

    rnd = acts.examples.RandomNumbers(seed=args.random_seed)
    outputDir = Path(outputDir)
    logger = acts.getDefaultLogger("GSF Example", acts.logging.INFO)

    os.makedirs(outputDir, exist_ok=True)

    if inputParticlePath is None:
        addParticleGun(
            s,
            ParticleConfig(num=1, pdg=acts.PdgParticle.eElectron, randomizeCharge=True),
            EtaConfig(-3.0, 3.0, uniform=True),
            MomentumConfig(1.0 * u.GeV, 100.0 * u.GeV, transverse=True),
            PhiConfig(0.0, 360.0 * u.degree),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                # stddev=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(0.015, 0.015, 55.0, 0),
            ),
            multiplicity=1,
            rnd=rnd,
            outputDirCsv=outputDir / "csv",
            outputDirRoot=outputDir / "root",
        )
    else:
        logger.info("Reading particles from {}", inputParticlePath.resolve())
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                outputParticles="particles_generated",
            )
        )
        s.addWhiteboardAlias("particles", "particles_generated")

    if inputSimHitsPath is None:
        if mode == "fatras":
            print("Running Fatras simulation")
            addFatras(
                s,
                trackingGeometry,
                field,
                rnd=rnd,
                enableInteractions=True,
            )
        else:
            print("Running Geant4 simulation")
            addGeant4(
                s,
                detector,
                trackingGeometry,
                field,
                outputDirCsv=outputDir / "geant4_csv",
                outputDirRoot=outputDir,
                outputDirObj=outputDir / "geant4_obj",
                rnd=rnd,
                materialMappings=["Silicon"],
                volumeMappings=[],
                killVolume=trackingGeometry.highestTrackingVolume,
                killAfterTime=25 * u.ns,
                recordHitsOfSecondaries=False,
                keepParticlesWithoutHits=False,
                killSecondaries=True,
            )
    else:
        logger.info("Reading hits from {}", inputSimHitsPath.resolve())
        s.addReader(
            RootSimHitReader(
                level=acts.logging.INFO,
                filePath=str(inputSimHitsPath.resolve()),
                outputSimHits="simhits",
            )
        )
        s.addWhiteboardAlias("particles_simulated_selected", "particles_generated")

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
        outputDirCsv=outputDir / "csv",
        outputDirRoot=outputDir / "root",
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(0.9 * u.GeV, None),
            measurements=(7, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        inputParticles="particles_generated",
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        trackSmearingSigmas=TrackSmearingSigmas(
            # zero everything so the GSF has a chance to find the measurements
            loc0=0,
            loc0PtA=0,
            loc0PtB=0,
            loc1=0,
            loc1PtA=0,
            loc1PtB=0,
            time=0,
            phi=0,
            theta=0,
            ptRel=0,
        ),
        particleHypothesis=acts.ParticleHypothesis.electron,
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0 / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
    )

    addTruthTrackingGsf(
        s,
        trackingGeometry,
        field,
    )

    s.addAlgorithm(
        acts.examples.TrackSelectorAlgorithm(
            level=acts.logging.INFO,
            inputTracks="tracks",
            # inputTracks="gsf_tracks",
            outputTracks="selected-tracks",
            selectorConfig=acts.TrackSelector.Config(
                minMeasurements=7,
            ),
        )
    )
    s.addWhiteboardAlias("tracks", "selected-tracks")
    # s.addWhiteboardAlias("gsf_selected_tracks", "selected-tracks")

    # WORKS HERE
    # s.addWriter(
    #     RootTrackStatesWriter(
    #         level=acts.logging.INFO,
    #         # inputTracks="tracks",
    #         inputTracks="gsf_tracks",
    #         inputParticles="particles_selected",
    #         inputTrackParticleMatching="track_particle_matching",
    #         inputSimHits="simhits",
    #         inputMeasurementSimHitsMap="measurement_simhits_map",
    #         # filePath=str(outputDir / "trackstates.root"),
    #         filePath=str(outputDir / "trackstates_gsf.root"),
    #     )
    # )

    # s.addWriter(
    #     RootTrackSummaryWriter(
    #         level=acts.logging.INFO,
    #         # inputTracks="tracks",
    #         inputTracks="gsf_tracks",
    #         inputParticles="particles_selected",
    #         inputTrackParticleMatching="track_particle_matching",
    #         # filePath=str(outputDir / "tracksummary.root"),
    #         filePath=str(outputDir / "tracksummary_gsf.root"),
    #         writeGsfSpecific=True,
    #     )
    # )

    s.addWriter(
        RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            # inputTracks="tracks",
            inputTracks="tracks",
            # inputTracks="gsf_tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            # filePath=str(outputDir / "performance.root"),
            filePath=str(outputDir / "performance_gsf.root"),
        )
    )

    gsfCfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    gsfCfg.inputTracks = "gsf_tracks"
    # cfg.inputParticles = "particles"
    # USE ONLY PARTICLES THAT SURVIVE DIGITIZATION / MEASUREMENT REQUIREMENTS
    gsfCfg.inputParticles = "particles_generated_selected"
    gsfCfg.inputTrackParticleMatching = "track_particle_matching"
    gsfCfg.inputParticleTrackMatching = "particle_track_matching"
    gsfCfg.inputParticleMeasurementsMap = "particle_measurements_map"
    gsfPerfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        # cfg, acts.logging.INFO
        gsfCfg,
        acts.logging.VERBOSE,
    )
    s.addWriter(gsfPerfWriter)

    return s, gsfPerfWriter


def runOddMlTrackFinding(
    trackingGeometry,
    field,
    digiConfigFile,
    geoSelectionConfigFile,
    stripGeoSelectionConfigFile,
    outputDir,
    mlModelFile,
    inputParticlePath: Optional[Path] = None,
    inputSimHitsPath: Optional[Path] = None,
    decorators=[],
    s=None,
):
    from acts.examples.simulation import (
        addParticleGun,
        ParticleConfig,
        EtaConfig,
        PhiConfig,
        MomentumConfig,
        addFatras,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
        addGeant4,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TrackSmearingSigmas,
        addTruthTrackingGsf,
    )
    from acts.examples.root import (
        RootTrackStatesWriter,
        RootTrackSummaryWriter,
        RootTrackFitterPerformanceWriter,
    )

    from regressor_models import MLP, printModelSummary

    if args.read_data:
        # Previous data scalers loaded here, will have to initialize with new directory to create new scalers
        dataHandler = DataHandler([dataDir], load_data_scalers=True)
        mlInputs = dataHandler.readX([dataDir])
        n_events = len(mlInputs)
    else:
        # n_events = 1
        n_events = 9999
        # n_events = 1000

    mode = "fatras"
    # mode = "geant4"
    s = s or acts.examples.Sequencer(
        events=n_events, numThreads=1, logLevel=acts.logging.INFO
    )
    # s = s or acts.examples.Sequencer(
    #     events=n_events,
    #     numThreads=1 if mode == "geant4" else -1,
    #     logLevel=acts.logging.INFO,
    # )

    for d in decorators:
        s.addContextDecorator(d)

    outputDir = outputDir / "electron"
    data_type = "train"
    if mode == "fatras":
        outputDir = outputDir / "fatras"
    else:
        outputDir = outputDir / "geant4"
    # print(outputDir)
    # print(type(outputDir))
    if n_events == 10000:
        dir_ending = ""
    else:
        dir_ending = "_{}".format(str(n_events))
    if data_type == "train":
        outputDir = outputDir / ("train" + dir_ending)
    else:
        outputDir = outputDir / ("test" + dir_ending)
    print(outputDir)
    print(type(outputDir))

    RANDOM_SEED = 42
    rnd = acts.examples.RandomNumbers(seed=RANDOM_SEED)
    outputDir = Path(outputDir)
    logger = acts.getDefaultLogger("GSF Example", acts.logging.INFO)

    os.makedirs(outputDir, exist_ok=True)

    if args.read_data:
        inputParticlePath = outputDir / "root" / "particles.root"
        inputSimHitsPath = outputDir / "hits.root"

    if inputParticlePath is None:
        # GENERIC DETECTOR EXAMPLE VARIANT
        # addParticleGun(
        #     s,
        #     MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
        #     EtaConfig(-2.0, 2.0, uniform=True),
        #     PhiConfig(0.0, 360.0 * u.degree),
        #     ParticleConfig(1, acts.PdgParticle.eMuon, randomizeCharge=True),
        #     rnd=rnd,
        #     # ADDED
        #     outputDirCsv=outputDir / "csv",
        #     outputDirRoot=outputDir / "root",
        # )
        # 2025 GSF VARIANT
        addParticleGun(
            s,
            ParticleConfig(num=1, pdg=acts.PdgParticle.eElectron, randomizeCharge=True),
            EtaConfig(-3.0, 3.0, uniform=True),
            MomentumConfig(1.0 * u.GeV, 100.0 * u.GeV, transverse=True),
            PhiConfig(0.0, 360.0 * u.degree),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                # stddev=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(0.015, 0.015, 55.0, 0),
            ),
            multiplicity=1,
            rnd=rnd,
            outputDirCsv=outputDir / "csv",
            outputDirRoot=outputDir / "root",
        )
    else:
        logger.info("Reading particles from {}", inputParticlePath.resolve())
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                outputParticles="particles_generated",
            )
        )
        # s.addWhiteboardAlias("particles", "particles_generated")
        s.addWhiteboardAlias("particles_generated_selected", "particles_generated")

    if inputSimHitsPath is None:
        if mode == "fatras":
            addFatras(
                s,
                trackingGeometry,
                field,
                rnd=rnd,
                # From the GSF example
                # enableInteractions=True,
                # From the generic detector example
                outputDirRoot=outputDir,
            )
        else:
            addGeant4(
                s,
                detector,
                trackingGeometry,
                field,
                outputDirCsv=outputDir / "geant4_csv",
                outputDirRoot=outputDir,
                outputDirObj=outputDir / "geant4_obj",
                rnd=rnd,
                materialMappings=["Silicon"],
                volumeMappings=[],
                killVolume=trackingGeometry.highestTrackingVolume,
                killAfterTime=25 * u.ns,
                recordHitsOfSecondaries=False,
                keepParticlesWithoutHits=False,
                killSecondaries=True,
            )
    else:
        logger.info("Reading hits from {}", inputSimHitsPath.resolve())
        s.addReader(
            RootSimHitReader(
                level=acts.logging.INFO,
                filePath=str(inputSimHitsPath.resolve()),
                outputSimHits="simhits",
            )
        )
        s.addWhiteboardAlias("particles_simulated_selected", "particles_generated")

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
        outputDirCsv=outputDir / "csv",
        outputDirRoot=outputDir / "root",
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(0.9 * u.GeV, None),
            measurements=(7, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    #####################
    # ML
    #####################
    s.addAlgorithm(
        acts.examples.SpacePointMaker(
            level=acts.logging.INFO,
            trackingGeometry=trackingGeometry,
            inputMeasurements="measurement_subset",
            outputSpacePoints="spacepoints",
            geometrySelection=acts.examples.json.readJsonGeometryList(
                str(geoSelectionConfigFile)
            ),
            stripGeometrySelection=acts.examples.json.readJsonGeometryList(
                str(stripGeoSelectionConfigFile)
            ),
        )
    )

    # NOT NEEDED?
    # trkParamExtractor = acts.examples.ParticleTrackParamExtractor(
    #     level=acts.logging.INFO,
    #     inputParticles="particles_generated_selected",
    #     outputTrackParameters="true_parameters",
    # )
    # s.addAlgorithm(trkParamExtractor)

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

            self.perigeeSurface = acts.Surface.createPerigee(
                acts.Vector3(0.0, 0.0, 0.0)
            )

            tech_acts_dir = (
                "/home/taleiko/Documents/CERN/Technical_Student/Program/acts"
            )
            train_data_dirs = [
                os.path.join(
                    tech_acts_dir,
                    "mega_data/mega_data_{}/{}/{}/train_100000".format(
                        str(num), "electron", "geant4"
                    ),
                )
                for num in range(10)
            ]

            # Glued together to work: Data directories not actually used here
            # NOTE: Now potential double data reading if new scalers are actually created
            # See earlier DataHandler
            dh = DataHandler(train_data_dirs, load_data_scalers=True)
            self.input_scaler = dh.getInputScaler()
            self.output_scaler = dh.getOutputScaler()

            self.max_seq_len = 20

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.mlp = MLP(input_dim=max_seq_len*3, output_dim=5, hidden_dim=32, n_hidden_layers=2)
            self.mlp = MLP(
                input_dim=self.max_seq_len * 3,
                output_dim=5,
                hidden_dim=256,
                n_hidden_layers=7,
            )
            self.mlp.to(device)
            self.mlp.load_state_dict(torch.load(mlModelFile, map_location=device))

        def execute(self, context):
            prototracks = self.prototracks(context.eventStore)
            spacepoints = self.spacepoints(context.eventStore)

            measurement_to_spacepoint = {}
            measurement_to_sourcelink = {}
            for sp in spacepoints:
                for sl in sp.sourceLinks:
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    meas_id = isl.index()
                    measurement_to_spacepoint[meas_id] = sp
                    measurement_to_sourcelink[meas_id] = sl

            container = acts.examples.TrackContainer()
            surface_map = trackingGeometry.geoIdSurfaceMap()
            # print(prototracks)

            # print(measurement_to_spacepoint)
            for prototrack in prototracks:
                # ml_input = np.array(
                #     [
                #         [
                #             measurement_to_spacepoint[meas_id].x,
                #             measurement_to_spacepoint[meas_id].y,
                #             measurement_to_spacepoint[meas_id].z,
                #         ]
                #         for meas_id in prototrack
                #     ]
                # )

                coords = []
                for meas_id in prototrack:
                    sp = measurement_to_spacepoint.get(meas_id)
                    if sp is None:
                        continue
                    coords.append([sp.x, sp.y, sp.z])
                # if len(coords) < 3:
                #     continue
                ml_input = np.array(coords)

                # print([meas_id for meas_id in prototrack])
                # print(ml_input)
                # sys.exit(0)

                # fig = plt.figure(figsize=(4, 4))
                # ax = fig.add_subplot(111, projection="3d")
                # for coord in ml_input:
                #     ax.scatter(coord[0], coord[1], coord[2])
                # # plt.show()
                # plt.savefig(
                #     "/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code/trajectory.png"
                # )

                # print(ml_input)

                scaled_input = self.input_scaler.transform(ml_input)
                # print(scaled_input)
                scaled_input = np.flip(scaled_input, axis=0)
                # print(scaled_input)
                pad_len = self.max_seq_len - len(scaled_input)
                scaled_input = np.pad(
                    scaled_input, ((0, pad_len), (0, 0)), mode="constant"
                )
                scaled_input = scaled_input.flatten()
                # print(scaled_input)
                scaled_input = torch.tensor(scaled_input, dtype=torch.float32)
                # print(scaled_input)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                scaled_input.to(device)

                # TODO: This is taped together at the moment. The scaler expects an array of columns. Note output[0] and array([output])
                with torch.no_grad():
                    scaled_output = np.array(
                        [self.mlp(scaled_input).detach().cpu().numpy()]
                    )
                output = self.output_scaler.inverse_transform(scaled_output)
                # print(output)
                output = output[0]
                # print(output)

                track = container.makeTrack()
                track.referenceSurface = self.perigeeSurface
                track.parameters = acts.BoundVector(
                    output[0], output[1], output[2], output[3], output[4], 1.0
                )
                track.particleHypothesis = acts.ParticleHypothesis.electron
                track.covariance = acts.BoundMatrix.Identity()
                track.chi2 = 0.0
                track.nHoles = 0
                track.nMeasurements = len(prototrack)

                # Attach measurements to the track state. Use the original source
                # link from the reconstructed space point and the matching
                # surface from the geometry map.
                for meas_id in prototrack:
                    # sp = measurement_to_spacepoint[meas_id]
                    sp = measurement_to_spacepoint.get(meas_id)
                    if sp is None:
                        continue
                    sl = measurement_to_sourcelink[meas_id]
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    sf = surface_map[isl.geometryId()]

                    trackState = track.appendTrackState()
                    trackState.typeFlags.isMeasurement = True
                    trackState.uncalibratedSourceLink = sl
                    trackState.referenceSurface = sf

            self.tracks(context, container.makeConst())
            return acts.examples.ProcessCode.SUCCESS

    s.addAlgorithm(PythonTrackFitter("PythonTrackFitter", acts.logging.INFO))

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            # level=acts.logging.INFO,
            level=acts.logging.VERBOSE,
            inputTracks="fitted_tracks",
            # inputParticles="particles",
            inputParticles="particles_generated_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="track_particle_matching",
            outputParticleTrackMatching="particle_track_matching",
            doubleMatching=True,
        )
    )

    # NOT NEEDED? FROM THE GSF SCRIPT
    # s.addAlgorithm(
    #     acts.examples.TrackSelectorAlgorithm(
    #         level=acts.logging.INFO,
    #         inputTracks="fitted_tracks",
    #         outputTracks="selected-tracks",
    #         selectorConfig=acts.TrackSelector.Config(
    #             minMeasurements=7,
    #         ),
    #     )
    # )
    # s.addWhiteboardAlias("tracks", "selected-tracks")

    cfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    cfg.inputTracks = "fitted_tracks"
    # cfg.inputParticles = "particles"
    # USE ONLY PARTICLES THAT SURVIVE DIGITIZATION / MEASUREMENT REQUIREMENTS
    cfg.inputParticles = "particles_generated_selected"
    cfg.inputTrackParticleMatching = "track_particle_matching"
    cfg.inputParticleTrackMatching = "particle_track_matching"
    cfg.inputParticleMeasurementsMap = "particle_measurements_map"
    perfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        # cfg, acts.logging.INFO
        cfg,
        acts.logging.VERBOSE,
    )
    s.addWriter(perfWriter)

    # Not working at the moment, need to extend the Python fitter/bindings to also fill calibrated measurements and projector information correctly
    # s.addWriter(
    #     RootTrackStatesWriter(
    #         level=acts.logging.INFO,
    #         # inputTracks="tracks",
    #         inputTracks="fitted_tracks",
    #         # inputParticles="particles_selected",
    #         # inputParticles="particles",
    #         inputParticles="particles_generated_selected",
    #         inputTrackParticleMatching="track_particle_matching",
    #         inputSimHits="simhits",
    #         inputMeasurementSimHitsMap="measurement_simhits_map",
    #         # filePath=str(outputDir / "trackstates.root"),
    #         filePath=str(outputDir / "trackstates_gsf.root"),
    #     )
    # )

    # s.addWriter(
    #     RootTrackSummaryWriter(
    #         level=acts.logging.INFO,
    #         # inputTracks="tracks",
    #         inputTracks="fitted_tracks",
    #         # inputParticles="particles_selected",
    #         # inputParticles="particles",
    #         inputParticles="particles_generated_selected",
    #         inputTrackParticleMatching="track_particle_matching",
    #         # filePath=str(outputDir / "tracksummary.root"),
    #         filePath=str(outputDir / "tracksummary_gsf.root"),
    #         writeGsfSpecific=True,
    #     )
    # )

    s.addWriter(
        RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            # inputTracks="tracks",
            inputTracks="fitted_tracks",
            # inputParticles="particles_selected",
            # inputParticles="particles",
            inputParticles="particles_generated_selected",
            inputTrackParticleMatching="track_particle_matching",
            # filePath=str(outputDir / "performance.root"),
            filePath=str(outputDir / "performance_ml.root"),
        )
    )

    return s, perfWriter


def runMlVsGsfTrackFinding(
    trackingGeometry,
    field,
    digiConfigFile,
    geoSelectionConfigFile,
    # stripGeoSelectionConfigFile,
    outputDir,
    mlModelFile,
    detector,
    inputParticlePath: Optional[Path] = None,
    inputSimHitsPath: Optional[Path] = None,
    decorators=[],
    s=None,
):
    from acts.examples.simulation import (
        addParticleGun,
        ParticleConfig,
        EtaConfig,
        PhiConfig,
        MomentumConfig,
        addFatras,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
        addGeant4,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TrackSmearingSigmas,
        addTruthTrackingGsf,
    )
    from acts.examples.root import (
        RootTrackStatesWriter,
        RootTrackSummaryWriter,
        RootTrackFitterPerformanceWriter,
    )

    from regressor_models import MLP, printModelSummary

    # mode = "fatras"
    # mode = "geant4"
    mode = args.mode

    n_events = args.n_events

    outputDir = outputDir / "electron"
    data_type = "train"
    if mode == "fatras":
        outputDir = outputDir / "fatras"
    else:
        outputDir = outputDir / "geant4"
    # print(outputDir)
    # print(type(outputDir))
    # if n_events == 10000:
    #     dir_ending = ""
    # else:
    dir_ending = "_{}".format(str(n_events))
    if data_type == "train":
        outputDir = outputDir / ("train" + dir_ending)
    else:
        outputDir = outputDir / ("test" + dir_ending)
    print(outputDir)
    print(type(outputDir))

    if args.read_data:
        inputParticlePath = outputDir / "root" / "particles.root"
        inputSimHitsPath = outputDir / "hits.root"

    # CONTINUE FROM HERE: PUT THE DATA PATH IN THE SIMHITS THING, THAT WILL AUTOMATE EVERYTHING DOWN THE PATH!!!!!
    dataDir = "/home/taleiko/Documents/CERN/Technical_Student/Program/acts/mega_data/mega_data_9/electron/geant4/train_100000"
    # if args.read_data:
    #     # Previous data scalers loaded here, will have to initialize with new directory to create new scalers
    #     dataHandler = DataHandler([dataDir], load_data_scalers=True)
    #     mlInputs = dataHandler.readX([dataDir])
    #     n_events = len(mlInputs)
    # else:
    #     # n_events = 1
    #     # n_events = 100
    #     # n_events = 1000
    #     # n_events = 10000
    #     # n_events = 100000
    #     n_events = args.n_events

    print(n_events)
    print(outputDir)
    # sys.exit(0)

    s = s or acts.examples.Sequencer(
        # events=n_events, numThreads=1, logLevel=acts.logging.INFO
        events=n_events,
        numThreads=1,
        logLevel=acts.logging.DEBUG,
    )

    for d in decorators:
        s.addContextDecorator(d)

    print("RANDOM SEED:", args.random_seed)
    rnd = acts.examples.RandomNumbers(seed=args.random_seed)
    outputDir = Path(outputDir)
    logger = acts.getDefaultLogger("GSF Example", acts.logging.INFO)

    os.makedirs(outputDir, exist_ok=True)

    if inputParticlePath is None:
        # GENERIC DETECTOR EXAMPLE VARIANT
        # addParticleGun(
        #     s,
        #     MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
        #     EtaConfig(-2.0, 2.0, uniform=True),
        #     PhiConfig(0.0, 360.0 * u.degree),
        #     ParticleConfig(1, acts.PdgParticle.eMuon, randomizeCharge=True),
        #     rnd=rnd,
        #     # ADDED
        #     outputDirCsv=outputDir / "csv",
        #     outputDirRoot=outputDir / "root",
        # )
        # 2025 GSF VARIANT
        addParticleGun(
            s,
            ParticleConfig(num=1, pdg=acts.PdgParticle.eElectron, randomizeCharge=True),
            EtaConfig(-3.0, 3.0, uniform=True),
            MomentumConfig(1.0 * u.GeV, 100.0 * u.GeV, transverse=True),
            PhiConfig(0.0, 360.0 * u.degree),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                # stddev=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(0.015, 0.015, 55.0, 0),
            ),
            multiplicity=1,
            rnd=rnd,
            outputDirCsv=outputDir / "csv",
            outputDirRoot=outputDir / "root",
        )
    else:
        logger.info("Reading particles from {}", inputParticlePath.resolve())
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                outputParticles="particles_generated",
            )
        )
        # s.addWhiteboardAlias("particles", "particles_generated")
        s.addWhiteboardAlias("particles_generated_selected", "particles_generated")

    if inputSimHitsPath is None:
        if mode == "fatras":
            addFatras(
                s,
                trackingGeometry,
                field,
                rnd=rnd,
                # 2025
                enableInteractions=True,
                outputDirRoot=outputDir,
            )
        else:
            addGeant4(
                s,
                detector,
                trackingGeometry,
                field,
                outputDirCsv=outputDir / "geant4_csv",
                outputDirRoot=outputDir,
                outputDirObj=outputDir / "geant4_obj",
                rnd=rnd,
                materialMappings=["Silicon"],
                volumeMappings=[],
                killVolume=trackingGeometry.highestTrackingVolume,
                killAfterTime=25 * u.ns,
                recordHitsOfSecondaries=False,
                keepParticlesWithoutHits=False,
                killSecondaries=True,
            )
    else:
        print(inputSimHitsPath)
        logger.info("Reading hits from {}", inputSimHitsPath.resolve())
        s.addReader(
            RootSimHitReader(
                level=acts.logging.INFO,
                filePath=str(inputSimHitsPath.resolve()),
                outputSimHits="simhits",
            )
        )
        s.addWhiteboardAlias("particles_simulated_selected", "particles_generated")

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
        outputDirCsv=outputDir / "csv",
        outputDirRoot=outputDir / "root",
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(0.9 * u.GeV, None),
            measurements=(7, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    ##################
    # GSF
    ##################

    # 2026 AI SUGGESTIONS
    #
    # trkParamExtractor = acts.examples.ParticleTrackParamExtractor(
    #     level=acts.logging.INFO,
    #     # inputParticles="particles_generated_selected",
    #     inputParticles="particles_selected",
    #     outputTrackParameters="trueparameters",
    # )
    # s.addAlgorithm(trkParamExtractor)

    # trkSmear = acts.examples.TrackParameterSmearing(
    #     level=acts.logging.INFO,
    #     inputTrackParameters="trueparameters",
    #     outputTrackParameters="estimatedparameters",
    #     randomNumbers=rnd,
    #     sigmaLoc0=0,
    #     sigmaLoc0PtA=0,
    #     sigmaLoc0PtB=0,
    #     sigmaLoc1=0,
    #     sigmaLoc1PtA=0,
    #     sigmaLoc1PtB=0,
    #     sigmaTime=0,
    #     sigmaPhi=0,
    #     sigmaTheta=0,
    #     sigmaPtRel=0,
    #     initialSigmas=[
    #         1 * u.mm,
    #         1 * u.mm,
    #         1 * u.degree,
    #         1 * u.degree,
    #         0 / u.GeV,
    #         1 * u.ns,
    #     ],
    #     initialSigmaQoverPt=0.1 / u.GeV,
    #     initialSigmaPtRel=0.1,
    #     initialVarInflation=[1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
    #     particleHypothesis=acts.ParticleHypothesis.electron,
    # )
    # s.addAlgorithm(trkSmear)

    # if inputSimHitsPath is not None:
    # addTruthTrackingGsf hard-codes inputParticles="particles" in its
    # internal truth matcher. When sim hits are read from disk, no
    # simulation algorithm creates that alias, so point it at the concrete
    # output of addDigiParticleSelection.
    # s.addWhiteboardAlias("particles", "tmp_particles_digitized_selected")
    s.addWhiteboardAlias("particles", "tmp_particles_digitized_selected")

    # 2026
    # addSeeding(
    #     s,
    #     trackingGeometry,
    #     field,
    #     rnd=rnd,
    #     inputParticles="particles_generated",
    #     seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
    #     trackSmearingSigmas=TrackSmearingSigmas(
    #         # zero everything so the GSF has a chance to find the measurements
    #         loc0=0,
    #         loc0PtA=0,
    #         loc0PtB=0,
    #         loc1=0,
    #         loc1PtA=0,
    #         loc1PtB=0,
    #         time=0,
    #         phi=0,
    #         theta=0,
    #         ptRel=0,
    #     ),
    #     particleHypothesis=acts.ParticleHypothesis.electron,
    #     initialSigmas=[
    #         1 * u.mm,
    #         1 * u.mm,
    #         1 * u.degree,
    #         1 * u.degree,
    #         0 / u.GeV,
    #         1 * u.ns,
    #     ],
    #     initialSigmaQoverPt=0.1 / u.GeV,
    #     initialSigmaPtRel=0.1,
    #     initialVarInflation=[1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
    # )
    # 2025
    addSeeding(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        inputParticles="particles_generated",
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        particleHypothesis=acts.ParticleHypothesis.electron,
    )

    addTruthTrackingGsf(
        s,
        trackingGeometry,
        field,
        inputProtoTracks="truth_particle_tracks",
        # Only <= 20 measurements
        # inputProtoTracks="filtered_truth_particle_tracks",
    )

    s.addAlgorithm(
        acts.examples.TrackSelectorAlgorithm(
            level=acts.logging.INFO,
            # inputTracks="tracks",
            inputTracks="gsf_tracks",
            outputTracks="selected-tracks",
            # outputTracks="selected_gsf_tracks",
            selectorConfig=acts.TrackSelector.Config(
                minMeasurements=7,
            ),
        )
    )
    # s.addWhiteboardAlias("tracks", "selected-tracks")
    s.addWhiteboardAlias("gsf_selected_tracks", "selected-tracks")

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=acts.logging.INFO,
            inputTracks="gsf_selected_tracks",
            inputParticles="particles_generated_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="gsf_selected_track_particle_matching",
            outputParticleTrackMatching="gsf_selected_particle_track_matching",
            doubleMatching=True,
        )
    )

    gsfCfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    gsfCfg.inputTracks = "gsf_tracks"
    # cfg.inputParticles = "particles"
    # USE ONLY PARTICLES THAT SURVIVE DIGITIZATION / MEASUREMENT REQUIREMENTS
    gsfCfg.inputParticles = "particles_generated_selected"
    gsfCfg.inputTrackParticleMatching = "gsf_selected_track_particle_matching"
    gsfCfg.inputParticleTrackMatching = "gsf_selected_particle_track_matching"
    gsfCfg.inputParticleMeasurementsMap = "particle_measurements_map"
    gsfPerfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        # cfg, acts.logging.INFO
        gsfCfg,
        acts.logging.VERBOSE,
    )
    s.addWriter(gsfPerfWriter)

    # resCfgGsf = acts.examples.root.ResPlotToolConfig()
    # resCfgGsf.varBinning["Residual_d0"] = acts.Axis.regular(
    #     200, -2.0, 2.0, "r_{d0} [mm]"
    # )
    s.addWriter(
        RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            # inputTracks="tracks",
            inputTracks="gsf_tracks",
            # inputParticles="particles_selected",
            inputParticles="particles_generated_selected",
            # inputTrackParticleMatching="track_particle_matching",
            inputTrackParticleMatching="gsf_selected_track_particle_matching",
            # filePath=str(outputDir / "performance.root"),
            filePath=str(outputDir / "performance_gsf.root"),
            # resPlotToolConfig=resCfgGsf,
        )
    )

    #####################
    # ML
    #####################
    s.addAlgorithm(
        acts.examples.SpacePointMaker(
            level=acts.logging.INFO,
            trackingGeometry=trackingGeometry,
            inputMeasurements="measurement_subset",
            outputSpacePoints="spacepoints",
            geometrySelection=acts.examples.json.readJsonGeometryList(
                str(geoSelectionConfigFile)
            ),
            # stripGeometrySelection=acts.examples.json.readJsonGeometryList(
            #     str(stripGeoSelectionConfigFile)
            # ),
        )
    )

    # NOT NEEDED?
    # trkParamExtractor = acts.examples.ParticleTrackParamExtractor(
    #     level=acts.logging.INFO,
    #     inputParticles="particles_generated_selected",
    #     outputTrackParameters="true_parameters",
    # )
    # s.addAlgorithm(trkParamExtractor)

    # ALREADY DONE IN addSeeding() chain of events. ML should be run after GSF
    # truthTrkFndAlg = acts.examples.TruthTrackFinder(
    #     level=acts.logging.INFO,
    #     # inputParticles="particles_generated_selected",
    #     inputParticles="particles_selected",
    #     inputMeasurements="measurements",
    #     inputParticleMeasurementsMap="particle_measurements_map",
    #     inputSimHits="simhits",
    #     inputMeasurementSimHitsMap="measurement_simhits_map",
    #     # outputProtoTracks="prototracks",
    #     outputProtoTracks="truth_particle_tracks",
    # )
    # s.addAlgorithm(truthTrkFndAlg)

    # # Filter prototracks to only keep those with <= 20 measurements (model constraint)
    # class ProtoTrackFilter(acts.examples.IAlgorithm):
    #     def __init__(self, name, level, max_measurements=20):
    #         acts.examples.IAlgorithm.__init__(self, name, level)
    #         self.max_measurements = max_measurements

    #         self.inputTracks = acts.examples.ReadDataHandle(
    #             self, acts.examples.ProtoTrackContainer, "Prototracks"
    #         )
    #         self.inputTracks.initialize("truth_particle_tracks")

    #         self.outputTracks = acts.examples.WriteDataHandle(
    #             self, acts.examples.ProtoTrackContainer, "Prototracks"
    #         )
    #         self.outputTracks.initialize("filtered_truth_particle_tracks")

    #     def execute(self, context):
    #         prototracks = self.inputTracks(context.eventStore)
    #         filtered_tracks = acts.examples.ProtoTrackContainer()

    #         for prototrack in prototracks:
    #             if len(prototrack) <= self.max_measurements:
    #                 filtered_tracks.append(prototrack)

    #         self.outputTracks(context, filtered_tracks)
    #         return acts.examples.ProcessCode.SUCCESS

    # s.addAlgorithm(ProtoTrackFilter("ProtoTrackFilter", acts.logging.INFO, max_measurements=20))

    class PythonTrackFitter(acts.examples.IAlgorithm):
        def __init__(self, name, level):
            acts.examples.IAlgorithm.__init__(self, name, level)

            self.prototracks = acts.examples.ReadDataHandle(
                self, acts.examples.ProtoTrackContainer, "Prototracks"
            )
            # self.prototracks.initialize("prototracks")
            self.prototracks.initialize("truth_particle_tracks")
            # Only <= 20 measurements
            # self.prototracks.initialize("filtered_truth_particle_tracks")

            self.tracks = acts.examples.WriteDataHandle(
                self, acts.examples.ConstTrackContainer, "Tracks"
            )
            self.tracks.initialize("fitted_tracks")

            # NEW
            self.spacepoints = acts.examples.ReadDataHandle(
                self, acts.SpacePointContainer2, "Spacepoints"
            )
            self.spacepoints.initialize("spacepoints")

            self.perigeeSurface = acts.Surface.createPerigee(
                acts.Vector3(0.0, 0.0, 0.0)
            )
            tech_acts_dir = (
                "/home/taleiko/Documents/CERN/Technical_Student/Program/acts"
            )
            train_data_dirs = [
                os.path.join(
                    tech_acts_dir,
                    "mega_data/mega_data_{}/{}/{}/train_100000".format(
                        str(num), "electron", "geant4"
                    ),
                )
                for num in range(10)
            ]
            # Glued together to work: Data directories not actually used here
            # NOTE: Now potential double data reading if new scalers are actually created
            # See earlier DataHandler
            self.dh = DataHandler(train_data_dirs, load_data_scalers=True)
            self.input_scaler = self.dh.getInputScaler()
            self.output_scaler = self.dh.getOutputScaler()

            self.max_seq_len = 20

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.mlp = MLP(input_dim=max_seq_len*3, output_dim=5, hidden_dim=32, n_hidden_layers=2)
            self.mlp = MLP(
                input_dim=self.max_seq_len * 3,
                output_dim=5,
                hidden_dim=256,
                n_hidden_layers=7,
            )
            self.mlp.to(device)
            self.mlp.load_state_dict(torch.load(mlModelFile, map_location=device))

        def execute(self, context):
            prototracks = self.prototracks(context.eventStore)
            spacepoints = self.spacepoints(context.eventStore)

            measurement_to_spacepoint = {}
            measurement_to_sourcelink = {}
            for sp in spacepoints:
                for sl in sp.sourceLinks:
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    meas_id = isl.index()
                    measurement_to_spacepoint[meas_id] = sp
                    measurement_to_sourcelink[meas_id] = sl

            container = acts.examples.TrackContainer()
            surface_map = trackingGeometry.geoIdSurfaceMap()
            # print(prototracks)

            # print(measurement_to_spacepoint)
            # logger.info("II'MMMM ALIIIIVEEEEEE!!!!!")
            # print(prototracks)
            # print([prototrack for prototrack in prototracks])
            # sys.exit(0)
            for prototrack in prototracks:
                coords = []
                for meas_id in prototrack:
                    # logger.info(str(meas_id))
                    sp = measurement_to_spacepoint.get(meas_id)
                    if sp is None:
                        # logger.info("PROOOOBLEEEEMMMM!!!!!")
                        # logger.info(str(sp))
                        continue
                    coords.append([sp.x, sp.y, sp.z])
                # if len(coords) < 3:
                #     continue
                if len(coords) > 20:
                    continue
                ml_input = np.array(coords)

                # print([meas_id for meas_id in prototrack])
                # print(ml_input)
                # sys.exit(0)

                # PLOT TRAJECTORY
                # fig = plt.figure(figsize=(4, 4))
                # ax = fig.add_subplot(111, projection="3d")
                # for coord in ml_input:
                #     ax.scatter(coord[0], coord[1], coord[2])
                # # plt.show()
                # plt.savefig(
                #     "/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code/trajectory.png"
                # )

                # print(ml_input)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scaled_input = self.input_scaler.transform(ml_input)
                # print(scaled_input)
                scaled_input = np.flip(scaled_input, axis=0)
                # print(scaled_input)
                # print(scaled_input.shape)
                # print(len(scaled_input))
                pad_len = self.max_seq_len - len(scaled_input)
                # print(self.max_seq_len)
                # print(pad_len)
                scaled_input = np.pad(
                    scaled_input, ((0, pad_len), (0, 0)), mode="constant"
                )
                scaled_input = scaled_input.flatten()
                # print(scaled_input)
                scaled_input = torch.tensor(scaled_input, dtype=torch.float32)
                # print(scaled_input)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                scaled_input.to(device)

                # TODO: This is taped together at the moment. The scaler expects an array of columns. Note output[0] and array([output])
                with torch.no_grad():
                    scaled_output = np.array(
                        [self.mlp(scaled_input).detach().cpu().numpy()]
                    )
                # print(scaled_output)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output = self.output_scaler.inverse_transform(scaled_output)
                # print(output)
                output = output[0]
                # print(output)

                track = container.makeTrack()
                track.referenceSurface = self.perigeeSurface
                track.parameters = acts.BoundVector(
                    output[0], output[1], output[2], output[3], output[4], 1.0
                )
                track.particleHypothesis = acts.ParticleHypothesis.electron
                track.covariance = acts.BoundMatrix.Identity()
                track.chi2 = 0.0
                track.nHoles = 0
                track.nMeasurements = len(prototrack)

                # Attach measurements to the track state. Use the original source
                # link from the reconstructed space point and the matching
                # surface from the geometry map.
                for meas_id in prototrack:
                    # sp = measurement_to_spacepoint[meas_id]
                    sp = measurement_to_spacepoint.get(meas_id)
                    if sp is None:
                        continue
                    sl = measurement_to_sourcelink[meas_id]
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    sf = surface_map[isl.geometryId()]

                    trackState = track.appendTrackState()
                    trackState.typeFlags.isMeasurement = True
                    trackState.uncalibratedSourceLink = sl
                    trackState.referenceSurface = sf

            self.tracks(context, container.makeConst())
            return acts.examples.ProcessCode.SUCCESS

    s.addAlgorithm(PythonTrackFitter("PythonTrackFitter", acts.logging.INFO))

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            # level=acts.logging.INFO,
            level=acts.logging.VERBOSE,
            inputTracks="fitted_tracks",
            # inputParticles="particles",
            # inputParticles="particles_generated_selected",
            inputParticles="particles_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="ml_track_particle_matching",
            outputParticleTrackMatching="ml_particle_track_matching",
            doubleMatching=True,
        )
    )

    mlCfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    mlCfg.inputTracks = "fitted_tracks"
    # cfg.inputParticles = "particles"
    # USE ONLY PARTICLES THAT SURVIVE DIGITIZATION / MEASUREMENT REQUIREMENTS
    mlCfg.inputParticles = "particles_generated_selected"
    mlCfg.inputTrackParticleMatching = "ml_track_particle_matching"
    mlCfg.inputParticleTrackMatching = "ml_particle_track_matching"
    mlCfg.inputParticleMeasurementsMap = "particle_measurements_map"
    mlPerfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        # cfg, acts.logging.INFO
        mlCfg,
        acts.logging.VERBOSE,
    )
    s.addWriter(mlPerfWriter)

    # resCfgMl = acts.examples.root.ResPlotToolConfig()
    # resCfgMl.varBinning["Residual_d0"] = acts.Axis.regular(
    #     100, -2.0, 2.0, "r_{d0} [mm]"
    # )
    # resCfgMl.varBinning["Residual_z0"] = acts.Axis.regular(
    #     100, -2.0, 2.0, "r_{z0} [mm]"
    # )
    # resCfgMl.varBinning["Residual_phi"] = acts.Axis.regular(
    #     100, -2.0, 2.0, "r_{phi} [rad]"
    # )
    # resCfgMl.varBinning["Residual_theta"] = acts.Axis.regular(
    #     100, -2.0, 2.0, "r_{theta} [rad]"
    # )
    # resCfgMl.varBinning["Residual_qop"] = acts.Axis.regular(
    #     100, -2.0, 2.0, "r_{q/p} [c/GeV]"
    # )
    s.addWriter(
        RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            # inputTracks="tracks",
            inputTracks="fitted_tracks",
            inputParticles="particles_selected",
            # inputParticles="particles_generated_selected",
            inputTrackParticleMatching="track_particle_matching",
            # filePath=str(outputDir / "performance.root"),
            filePath=str(outputDir / "performance_ml.root"),
            # resPlotToolConfig=resCfgMl,
        )
    )

    return s, mlPerfWriter, gsfPerfWriter


def runMlPredictionsFromOddRootData(
    trackingGeometry,
    field,
    digiConfigFile,
    geoSelectionConfigFile,
    outputDir,
    mlModelFile,
    dataDir,
    decorators=[],
    s=None,
):
    from acts.examples.root import (
        RootTrackSummaryWriter,
    )
    from acts.examples.reconstruction import addTruthTrackingGsf
    from acts.examples.simulation import (
        addDigitization,
        addDigiParticleSelection,
        ParticleSelectorConfig,
    )
    from regressor_models import MLP

    particleFile = dataDir / "root" / "particles.root"
    if not particleFile.exists():
        raise FileNotFoundError(f"Could not find particle ROOT file '{particleFile}'. ")

    simHitsFile = dataDir / "hits.root"
    if not simHitsFile.exists():
        raise FileNotFoundError(f"Could not find sim-hit ROOT file '{simHitsFile}'. ")

    # Bottleneck: Data potentially read twice if data scalers are not loaded
    dh = DataHandler([dataDir], load_data_scalers=True)
    mlInputs = dh.readX([dataDir])
    print(len(mlInputs))
    sys.exit(0)
    outputScaler = dh.getOutputScaler()

    nEvents = len(mlInputs)
    s = s or acts.examples.Sequencer(
        events=nEvents, numThreads=1, logLevel=acts.logging.INFO
    )
    rnd = acts.examples.RandomNumbers(seed=42)

    for d in decorators:
        s.addContextDecorator(d)

    s.addReader(
        RootParticleReader(
            level=acts.logging.INFO,
            filePath=str(particleFile.resolve()),
            outputParticles="particles_generated",
        )
    )
    s.addReader(
        RootSimHitReader(
            level=acts.logging.INFO,
            filePath=str(simHitsFile.resolve()),
            outputSimHits="simhits",
        )
    )
    s.addWhiteboardAlias("particles_simulated_selected", "particles_generated")

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        outputDirRoot=outputDir if args.output_root else None,
        outputDirCsv=outputDir if args.output_csv else None,
        rnd=rnd,
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(0.9 * u.GeV, None),
            measurements=(7, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
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

    truthTrkFndAlg = acts.examples.TruthTrackFinder(
        level=acts.logging.INFO,
        inputParticles="particles_selected",
        inputMeasurements="measurements",
        inputParticleMeasurementsMap="particle_measurements_map",
        inputSimHits="simhits",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputProtoTracks="truth_particle_tracks",
    )
    s.addAlgorithm(truthTrkFndAlg)

    # The GSF needs initial track parameters. For this comparison we seed it from
    # truth with zero smearing, matching truth_tracking_gsf.py.
    trkParamExtractor = acts.examples.ParticleTrackParamExtractor(
        level=acts.logging.INFO,
        inputParticles="particles_selected",
        outputTrackParameters="trueparameters",
    )
    s.addAlgorithm(trkParamExtractor)

    trkSmear = acts.examples.TrackParameterSmearing(
        level=acts.logging.INFO,
        inputTrackParameters="trueparameters",
        outputTrackParameters="estimatedparameters",
        randomNumbers=rnd,
        sigmaLoc0=0,
        sigmaLoc0PtA=0,
        sigmaLoc0PtB=0,
        sigmaLoc1=0,
        sigmaLoc1PtA=0,
        sigmaLoc1PtB=0,
        sigmaTime=0,
        sigmaPhi=0,
        sigmaTheta=0,
        sigmaPtRel=0,
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0 / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
        particleHypothesis=acts.ParticleHypothesis.electron,
    )
    s.addAlgorithm(trkSmear)

    # addTruthTrackingGsf hard-codes inputParticles="particles" in its internal
    # truth matcher. Alias to the concrete ParticleSelector output; aliasing to
    # "particles_selected" is an alias-to-alias and is not resolved here.
    s.addWhiteboardAlias("particles", "tmp_particles_digitized_selected")
    addTruthTrackingGsf(
        s,
        trackingGeometry,
        field,
        inputProtoTracks="truth_particle_tracks",
        logLevel=acts.logging.INFO,
    )

    s.addAlgorithm(
        acts.examples.TrackSelectorAlgorithm(
            level=acts.logging.INFO,
            inputTracks="gsf_tracks",
            outputTracks="selected_gsf_tracks",
            selectorConfig=acts.TrackSelector.Config(
                minMeasurements=7,
            ),
        )
    )

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=acts.logging.INFO,
            inputTracks="selected_gsf_tracks",
            inputParticles="particles_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="selected_gsf_track_particle_matching",
            outputParticleTrackMatching="selected_gsf_particle_track_matching",
            doubleMatching=True,
        )
    )

    class MlRootTrackFitter(acts.examples.IAlgorithm):
        def __init__(self, name, level):
            acts.examples.IAlgorithm.__init__(self, name, level)

            self.prototracks = acts.examples.ReadDataHandle(
                self, acts.examples.ProtoTrackContainer, "Prototracks"
            )
            self.prototracks.initialize("truth_particle_tracks")

            self.spacepoints = acts.examples.ReadDataHandle(
                self, acts.SpacePointContainer2, "Spacepoints"
            )
            self.spacepoints.initialize("spacepoints")

            self.tracks = acts.examples.WriteDataHandle(
                self, acts.examples.ConstTrackContainer, "Tracks"
            )
            self.tracks.initialize("ml_tracks")

            self.perigeeSurface = acts.Surface.createPerigee(
                acts.Vector3(0.0, 0.0, 0.0)
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            self.mlp = MLP(
                input_dim=mlInputs.shape[1],
                output_dim=5,
                hidden_dim=256,
                n_hidden_layers=7,
            )
            self.mlp.to(device)
            self.mlp.load_state_dict(torch.load(mlModelFile, map_location=device))
            self.mlp.eval()

        def execute(self, context):
            eventNumber = context.eventNumber
            if eventNumber >= len(mlInputs):
                raise RuntimeError(
                    f"No ML input available for event {eventNumber}; "
                    f"readX() returned {len(mlInputs)} events"
                )

            prototracks = self.prototracks(context.eventStore)
            spacepoints = self.spacepoints(context.eventStore)

            measurement_to_sourcelink = {}
            for sp in spacepoints:
                for sl in sp.sourceLinks:
                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    measurement_to_sourcelink[isl.index()] = sl
            surface_map = trackingGeometry.geoIdSurfaceMap()

            scaled_input = torch.tensor(
                mlInputs[eventNumber], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                scaled_output = self.mlp(scaled_input).detach().cpu().numpy()
            output = outputScaler.inverse_transform(scaled_output)[0]

            container = acts.examples.TrackContainer()

            for prototrack in prototracks:
                track = container.makeTrack()
                track.referenceSurface = self.perigeeSurface
                track.parameters = acts.BoundVector(
                    output[0], output[1], output[2], output[3], output[4], 0.0
                )
                track.particleHypothesis = acts.ParticleHypothesis.electron
                track.chi2 = 0.0

                nMeasurements = 0
                for meas_id in prototrack:
                    sl = measurement_to_sourcelink.get(meas_id)
                    if sl is None:
                        continue

                    isl = acts.examples.IndexSourceLink.FromSourceLink(sl)
                    sf = surface_map[isl.geometryId()]

                    trackState = track.appendTrackState()
                    trackState.typeFlags.isMeasurement = True
                    trackState.uncalibratedSourceLink = sl
                    trackState.referenceSurface = sf
                    nMeasurements += 1

                track.nMeasurements = nMeasurements

            self.tracks(context, container.makeConst())
            return acts.examples.ProcessCode.SUCCESS

    s.addAlgorithm(MlRootTrackFitter("MlRootTrackFitter", acts.logging.INFO))

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=acts.logging.INFO,
            inputTracks="ml_tracks",
            inputParticles="particles_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="ml_track_particle_matching",
            outputParticleTrackMatching="ml_particle_track_matching",
            doubleMatching=True,
        )
    )

    cfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    cfg.inputTracks = "ml_tracks"
    cfg.inputParticles = "particles_selected"
    cfg.inputTrackParticleMatching = "ml_track_particle_matching"
    cfg.inputParticleTrackMatching = "ml_particle_track_matching"
    cfg.inputParticleMeasurementsMap = "particle_measurements_map"
    perfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        cfg,
        acts.logging.INFO,
    )
    s.addWriter(perfWriter)

    gsfCfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    gsfCfg.inputTracks = "selected_gsf_tracks"
    gsfCfg.inputParticles = "particles_selected"
    gsfCfg.inputTrackParticleMatching = "selected_gsf_track_particle_matching"
    gsfCfg.inputParticleTrackMatching = "selected_gsf_particle_track_matching"
    gsfCfg.inputParticleMeasurementsMap = "particle_measurements_map"
    gsfPerfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        gsfCfg,
        acts.logging.INFO,
    )
    s.addWriter(gsfPerfWriter)

    mlSummaryPath = outputDir / "tracksummary_ml.root"
    gsfSummaryPath = outputDir / "tracksummary_gsf.root"
    s.addWriter(
        RootTrackSummaryWriter(
            level=acts.logging.INFO,
            inputTracks="ml_tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="ml_track_particle_matching",
            filePath=str(mlSummaryPath),
            treeName="tracksummary",
        )
    )
    s.addWriter(
        RootTrackSummaryWriter(
            level=acts.logging.INFO,
            inputTracks="selected_gsf_tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="selected_gsf_track_particle_matching",
            filePath=str(gsfSummaryPath),
            treeName="tracksummary",
            writeGsfSpecific=True,
        )
    )

    return s, perfWriter, gsfPerfWriter, mlSummaryPath, gsfSummaryPath
    # return s, perfWriter


if __name__ == "__main__":
    # srcdir = Path(__file__).resolve().parent.parent.parent.parent
    # srcdir = Path(__file__).resolve().parent.parent / "Technical_student" / "Program" / "acts"
    # srcdir = Path(__file__).resolve() / "acts"
    # srcdir = Path(__file__).resolve().parent
    actsSrcDir = Path(__file__).resolve().parent.parent.parent.parent
    phdSrcDir = Path("/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code")
    # print(srcdir)
    # sys.exit(0)

    if args.odd:
        # geoDir = getOpenDataDetectorDirectory()
        # oddMaterialMap = (
        #     args.material_config
        #     if args.material_config
        #     else geoDir / "data/odd-material-maps.root"
        # )
        # oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
        # detector = getOpenDataDetector(
        #     odd_dir=geoDir, materialDecorator=oddMaterialDeco
        # )
        # digiConfigFile = (
        #     getOpenDataDetectorDirectory() / "config/odd-digi-smearing-config.json"
        # )
        # SHOULD NOT BE USED?
        stripGeoSelectionConfigFile = (
            actsSrcDir / "Examples/Configs/odd-strip-spacepoint-selection.json"
        )
        # oddSeedingSel = actsSrcDir / "Examples/Configs/odd-seeding-config.json"
        oddSeedingSel = actsSrcDir / "Examples/Configs/odd-seeding-config_2026.json"
        geoSelectionConfigFile = oddSeedingSel

        detector = getOpenDataDetector()
        digiConfigFile = (
            # getOpenDataDetectorDirectory() / "config/odd-digi-smearing-config.json"
            actsSrcDir
            # / "Examples/Configs/odd-digi-smearing-config.json"
            / "Examples/Configs/odd-digi-smearing-config_2025.json"
        )
    else:
        detector = acts.examples.GenericDetector(acts.examples.GenericDetector.Config())
        digiConfigFile = (
            # actsSrcDir / "Examples/Configs/generic-digi-smearing-config.json"
            phdSrcDir
            / "generic-digi-smearing-config.json"
        )
        geoSelectionConfigFile = (
            phdSrcDir / "generic-pixel-sstrips-lstrips-spacepoints.json"
        )
    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()

    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))

    mlModelFile = "/home/taleiko/Documents/CERN/Technical_Student/Resultat/mega_mlp_1000e_8h_256n_0.001lr_1024b/mega_mlp_1000e_8h_256n_0.001lr_1024b.pt"
    # dataDirs = ["/home/taleiko/Documents/CERN/Doktorsstudier/Program/acts/test_data/test_data_0/electron/geant4/train_1"]
    dataDir = Path(
        "/home/taleiko/Documents/CERN/Doktorsstudier/Program/acts/test_data/test_data_0/electron/geant4/train_1"
    )
    # dataDir = Path("/home/taleiko/Documents/CERN/Doktorsstudier/Program/acts/test_data/test_data_1/electron/geant4/train_2")

    # outputDir = Path.cwd() / "output_track_finding_python_only"
    # outputDir = Path.cwd() / "output_track_finding_python_only" / "mega_data_9"
    # outputDir = Path.cwd() / "output_track_finding_python_only" / "mega_data_96"
    outputDir = Path.cwd() / "output_track_finding_python_only" / "mini_data_9"
    outputDir.mkdir(exist_ok=True)

    if args.read_data:
        inputParticlePath = "-1"
        inputSimHitsPath = "-1"
    else:
        inputParticlePath = None
        inputSimHitsPath = None

    # Simulate data on the go...
    # s, perfWriter = runTrackFindingPythonOnly(
    #     trackingGeometry=trackingGeometry,
    #     field=field,
    #     digiConfigFile=digiConfigFile,
    #     geoSelectionConfigFile=geoSelectionConfigFile,
    #     outputDir=outputDir,
    #     mlModelFile=mlModelFile,
    #     inputParticlePath=inputParticlePath,
    #     inputSimHitsPath=inputSimHitsPath,
    #     decorators=decorators,
    # )
    # s.run()
    # s, perfWriter = runOddTrackFinding(
    # s, performanceWriter = runOddGsfTrackFinding(
    #     trackingGeometry=trackingGeometry,
    #     field=field,
    #     digiConfigFile=digiConfigFile,
    #     geoSelectionConfigFile=geoSelectionConfigFile,
    #     stripGeoSelectionConfigFile=stripGeoSelectionConfigFile,
    #     outputDir=outputDir,
    #     mlModelFile=mlModelFile,
    #     inputParticlePath=inputParticlePath,
    #     inputSimHitsPath=inputSimHitsPath,
    #     decorators=decorators,
    # )
    # s, mlPerfWriter = runOddMlTrackFinding(
    #     trackingGeometry=trackingGeometry,
    #     field=field,
    #     digiConfigFile=digiConfigFile,
    #     geoSelectionConfigFile=geoSelectionConfigFile,
    #     stripGeoSelectionConfigFile=stripGeoSelectionConfigFile,
    #     outputDir=outputDir,
    #     mlModelFile=mlModelFile,
    #     inputParticlePath=inputParticlePath,
    #     inputSimHitsPath=inputSimHitsPath,
    #     decorators=decorators,
    # )
    s, mlPerfWriter, gsfPerfWriter = runMlVsGsfTrackFinding(
        trackingGeometry=trackingGeometry,
        field=field,
        digiConfigFile=digiConfigFile,
        geoSelectionConfigFile=geoSelectionConfigFile,
        # stripGeoSelectionConfigFile=stripGeoSelectionConfigFile,
        outputDir=outputDir,
        mlModelFile=mlModelFile,
        detector=detector,
        inputParticlePath=inputParticlePath,
        inputSimHitsPath=inputSimHitsPath,
        decorators=decorators,
    )
    # ...or read simulated data from ROOT files
    # s, perfWriter, gsfPerfWriter, mlSummaryPath, gsfSummaryPath = (
    #     runMlPredictionsFromOddRootData(
    #         trackingGeometry=trackingGeometry,
    #         field=field,
    #         digiConfigFile=digiConfigFile,
    #         geoSelectionConfigFile=geoSelectionConfigFile,
    #         outputDir=outputDir,
    #         mlModelFile=mlModelFile,
    #         dataDir=dataDir,
    #         decorators=decorators,
    #     )
    # )

    s.run()
    # sys.exit(0)

    print(mlPerfWriter.histograms().keys())
    fig, ax = plt.subplots()
    # print(type(histWriter.histograms()['trackeff_vs_eta'].plot(ax=ax)))
    # performanceWriter.histograms()["trackeff_vs_eta"].plot(ax=ax)
    mlPerfWriter.histograms()["trackeff_vs_eta"].plot(ax=ax)
    # sys.exit(0)
    # histWriter.histograms()['trackeff_vs_eta'].plot(ax=ax)
    # plt.show()
    # ax.set_xlim(-0.1, 0.1)
    plt.savefig(
        # "/home/taleiko/Documents/CERN/Doktorsstudier/Program/phd_code/ml_hist.png"
        outputDir
        / "ml_hist.png"
    )
    sys.exit(0)

    fig, ax = plt.subplots()
    gsfPerfWriter.histograms()["trackeff_vs_eta"].plot(ax=ax)
    plt.savefig(outputDir / "gsf_hist.png")

    fig, ax = plt.subplots()
    mlPerfWriter.histograms()["trackeff_vs_eta"].plot(ax=ax)
    plt.savefig(outputDir / "gsf_trackeff_vs_eta.png")

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
