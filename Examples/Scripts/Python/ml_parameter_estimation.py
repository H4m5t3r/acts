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
from typing import List, Optional
import argparse
import warnings

os.environ["ACTS_SEQUENCER_DISABLE_FPEMON"] = "1"

import acts
import acts.examples
from acts import UnitConstants as u
from acts.examples.odd import getOpenDataDetector
from acts.examples.root import (
    RootParticleReader,
    RootSimHitReader,
)

import numpy as np
import torch

import matplotlib.pyplot as plt

from ml_utilities import (
    DataHandler,
)

parser = argparse.ArgumentParser()
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
    "--n-events",
    help="Set the number of events to be simulated if data is not read",
    default=1,
    type=int,
)
parser.add_argument(
    "--mode",
    help="Set the simulation mode",
    choices=["fatras", "geant4"],
    default="geant4",
    type=str,
)
parser.add_argument(
    "--odd",
    help="Switch use of the ODD on/off",
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
    "--random-seed",
    help="Set the random seed for data generation",
    default=42,
    type=int,
)
args = parser.parse_args()


def runMlTrackFinding(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    digiConfigFile: Path,
    geoSelectionConfigFile: Path,
    outputDir: Path,
    mlModelFile: Path,
    detector: acts.examples.DetectorBase,
    inputParticlePath: Optional[Path] = None,
    inputSimHitsPath: Optional[Path] = None,
    decorators=[],
    s=None,
    train_data_dirs=List[Path],
    input_scaler_path: Optional[Path] = None,
    output_scaler_path: Optional[Path] = None,
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
        addTruthTrackingGsf,
    )
    from acts.examples.root import (
        RootTrackFitterPerformanceWriter,
    )

    from regressor_models import MLP, printModelSummary

    s = s or acts.examples.Sequencer(
        events=args.n_events, numThreads=-1, logLevel=acts.logging.INFO
    )

    for d in decorators:
        s.addContextDecorator(d)

    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)
    logger = acts.getDefaultLogger("ML Example", acts.logging.INFO)

    if inputParticlePath is None:
        addParticleGun(
            s,
            ParticleConfig(num=1, pdg=acts.PdgParticle.eElectron, randomizeCharge=True),
            EtaConfig(-3.0, 3.0, uniform=True),
            MomentumConfig(1.0 * u.GeV, 100.0 * u.GeV, transverse=True),
            PhiConfig(0.0, 360.0 * u.degree),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(0.015, 0.015, 55.0, 0),
            ),
            multiplicity=1,
            rnd=rnd,
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
        s.addWhiteboardAlias("particles_generated_selected", "particles_generated")

    if inputSimHitsPath is None:
        if args.mode == "fatras":
            addFatras(
                s,
                trackingGeometry,
                field,
                rnd=rnd,
                enableInteractions=True,
                outputDirCsv=outputDir / "csv",
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
        outputDirRoot=outputDir,
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
    # GSF (reference)
    #####################

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
    )

    s.addAlgorithm(
        acts.examples.TrackSelectorAlgorithm(
            level=acts.logging.INFO,
            inputTracks="gsf_tracks",
            outputTracks="selected-tracks",
            selectorConfig=acts.TrackSelector.Config(
                minMeasurements=7,
            ),
        )
    )
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
    gsfCfg.inputParticles = "particles_generated_selected"
    gsfCfg.inputTrackParticleMatching = "gsf_selected_track_particle_matching"
    gsfCfg.inputParticleTrackMatching = "gsf_selected_particle_track_matching"
    gsfCfg.inputParticleMeasurementsMap = "particle_measurements_map"
    gsfPerfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        gsfCfg, acts.logging.INFO
    )
    s.addWriter(gsfPerfWriter)

    s.addWriter(
        RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            inputTracks="gsf_tracks",
            inputParticles="particles_generated_selected",
            inputTrackParticleMatching="gsf_selected_track_particle_matching",
            filePath=str(outputDir / "performance_gsf.root"),
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
        )
    )

    class MLTrackFitter(acts.examples.IAlgorithm):
        def __init__(self, name, level):
            acts.examples.IAlgorithm.__init__(self, name, level)

            self.prototracks = acts.examples.ReadDataHandle(
                self, acts.examples.ProtoTrackContainer, "Prototracks"
            )
            self.prototracks.initialize("truth_particle_tracks")

            self.tracks = acts.examples.WriteDataHandle(
                self, acts.examples.ConstTrackContainer, "Tracks"
            )
            self.tracks.initialize("fitted_tracks")

            self.spacepoints = acts.examples.ReadDataHandle(
                self, acts.SpacePointContainer2, "Spacepoints"
            )
            self.spacepoints.initialize("spacepoints")

            self.perigeeSurface = acts.Surface.createPerigee(
                acts.Vector3(0.0, 0.0, 0.0)
            )
            self.dh = DataHandler(
                train_data_dirs=train_data_dirs,
                load_data_scalers=True,
                input_scaler_path=input_scaler_path,
                output_scaler_path=output_scaler_path,
            )
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
            for prototrack in prototracks:
                coords = []
                for meas_id in prototrack:
                    sp = measurement_to_spacepoint.get(meas_id)
                    if sp is None:
                        continue
                    coords.append([sp.x, sp.y, sp.z])
                # Example MLP trained with max 20 3D measurements as inputs
                if len(coords) > 20:
                    continue
                ml_input = np.array(coords)

                # TODO: Scaler was trained with feature names from Pandas DataFrames
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scaled_input = self.input_scaler.transform(ml_input)
                scaled_input = np.flip(scaled_input, axis=0)
                pad_len = self.max_seq_len - len(scaled_input)
                scaled_input = np.pad(
                    scaled_input, ((0, pad_len), (0, 0)), mode="constant"
                )
                scaled_input = scaled_input.flatten()
                scaled_input = torch.tensor(scaled_input, dtype=torch.float32)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                scaled_input.to(device)

                # TODO: This is taped together at the moment. The scaler expects an array of columns. Note output[0] and array([output])
                # For individual events there is only one set of outputs
                with torch.no_grad():
                    scaled_output = np.array(
                        [self.mlp(scaled_input).detach().cpu().numpy()]
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output = self.output_scaler.inverse_transform(scaled_output)
                output = output[0]

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

                for meas_id in prototrack:
                    # sp = measurement_to_spacepoint[meas_id]
                    sp = measurement_to_spacepoint.get(meas_id)
                    # TODO: Is this still a valid case?
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

    s.addAlgorithm(MLTrackFitter("MLTrackFitter", acts.logging.INFO))

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=acts.logging.INFO,
            inputTracks="fitted_tracks",
            inputParticles="particles_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching="ml_track_particle_matching",
            outputParticleTrackMatching="ml_particle_track_matching",
            doubleMatching=True,
        )
    )

    mlCfg = acts.examples.PythonTrackFinderPerformanceWriter.Config()
    mlCfg.inputTracks = "fitted_tracks"
    mlCfg.inputParticles = "particles_generated_selected"
    mlCfg.inputTrackParticleMatching = "ml_track_particle_matching"
    mlCfg.inputParticleTrackMatching = "ml_particle_track_matching"
    mlCfg.inputParticleMeasurementsMap = "particle_measurements_map"
    mlPerfWriter = acts.examples.PythonTrackFinderPerformanceWriter(
        mlCfg, acts.logging.INFO
    )
    s.addWriter(mlPerfWriter)

    s.addWriter(
        RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            inputTracks="fitted_tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            filePath=str(outputDir / "performance_ml.root"),
        )
    )

    return s, mlPerfWriter, gsfPerfWriter


if __name__ == "__main__":
    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    if args.odd:
        detector = getOpenDataDetector()
        digiConfigFile = srcdir / "Examples/Configs/odd-digi-smearing-config_2025.json"
        geoSelectionConfigFile = (
            srcdir / "Examples/Configs/odd-seeding-config_2026.json"
        )
    else:
        detector = acts.examples.GenericDetector(acts.examples.GenericDetector.Config())
        digiConfigFile = srcdir / "Examples/Configs/generic-digi-smearing-config.json"
        geoSelectionConfigFile = (
            srcdir / "Examples/Configs/generic-pixel-sstrips-lstrips-spacepoints.json"
        )

    trackingGeometry = detector.trackingGeometry()
    decorators = detector.contextDecorators()
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))

    outputDir = Path.cwd() / "output_ml" / "inference"
    mlModelFile = Path(
        "/home/taleiko/Documents/CERN/Doktorsstudier/Program/acts/ml_plugins"
    )

    if args.read_data:
        inputParticlePath = outputDir / "particles.root"
        inputSimHitsPath = outputDir / "hits.root"
    else:
        inputParticlePath = None
        inputSimHitsPath = None

    #######################################################
    # REMOVE IN SHIPPING
    input_scaler_path = "/home/taleiko/Documents/CERN/Technical_Student/Program/ml_model/input_scaler.pkl"
    output_scaler_path = "/home/taleiko/Documents/CERN/Technical_Student/Program/ml_model/output_scaler.pkl"
    tech_acts_dir = "/home/taleiko/Documents/CERN/Technical_Student/Program/acts"
    train_data_dirs = [
        Path(
            os.path.join(
                srcdir,
                "ml_data/mega_data_{}".format(str(num)),
            )
            for num in [0, 1]
        )
    ]
    #######################################################

    s, mlPerfWriter, gsfPerfWriter = runMlTrackFinding(
        trackingGeometry=trackingGeometry,
        field=field,
        digiConfigFile=digiConfigFile,
        geoSelectionConfigFile=geoSelectionConfigFile,
        outputDir=outputDir,
        mlModelFile=mlModelFile,
        detector=detector,
        inputParticlePath=inputParticlePath,
        inputSimHitsPath=inputSimHitsPath,
        decorators=decorators,
        train_data_dirs=train_data_dirs,
        input_scaler_path=input_scaler_path,
        output_scaler_path=output_scaler_path,
    )
