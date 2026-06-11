#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import acts
import acts.examples
from acts.examples.root import (
    RootParticleReader,
    RootSimHitReader,
)

u = acts.UnitConstants

# ADDED
import os
import argparse

print("Process ID:", os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=("fatras", "geant4"),
    default="fatras",
    help='Set the simulation type to "fatras" or "geant4"',
)
parser.add_argument(
    "--data_type",
    choices=("train", "test"),
    default="train",
    help='Define whether the output should be put in a directory called "train" or "test"',
)
parser.add_argument(
    "--n_events",
    type=int,
    default=10000,
    help="Set the number of events to be simulated",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path.cwd() / "ml_output",
    help="Set a custom output directory",
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
    help="Set the random seed for the simulation",
)

args = parser.parse_args()
MODE = args.mode
print("Using mode", MODE)
DATA_TYPE = args.data_type
print("Creating data of type", DATA_TYPE)
N_EVENTS = args.n_events
print("Simulating", N_EVENTS, "events")
OUTPUT_DIR = args.output_dir / "electron"
print("Using default output directory:", OUTPUT_DIR)
RANDOM_SEED = args.random_seed
print("Using random seed:", RANDOM_SEED)


def runTruthTrackingGsf(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    digiConfigFile: Path,
    outputDir: Path,
    inputParticlePath: Optional[Path] = None,
    inputSimHitsPath: Optional[Path] = None,
    decorators=[],
    s: acts.examples.Sequencer = None,
    mode: str = "fatras",
    data_type: str = "train",
    n_events: int = 10000,
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

    if mode not in ("fatras", "geant4"):
        raise ValueError(
            f"Unknown simulation mode '{mode}', expected 'fatras' or 'geant4'"
        )

    s = s or acts.examples.Sequencer(
        events=n_events,
        numThreads=1 if mode == "geant4" else -1,
        logLevel=acts.logging.INFO,
    )

    for d in decorators:
        s.addContextDecorator(d)

    print(outputDir)
    print(type(outputDir))
    # ADDED
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

    # rnd = acts.examples.RandomNumbers(seed=42)
    # rnd = acts.examples.RandomNumbers()
    rnd = acts.examples.RandomNumbers(seed=RANDOM_SEED)
    outputDir = Path(outputDir)
    logger = acts.getDefaultLogger("GSF Example", acts.logging.INFO)

    os.makedirs(outputDir)
    # outputDir.mkdir(exist_ok=True)

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
            outputTracks="selected-tracks",
            selectorConfig=acts.TrackSelector.Config(
                minMeasurements=7,
            ),
        )
    )
    s.addWhiteboardAlias("tracks", "selected-tracks")

    s.addWriter(
        RootTrackStatesWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            inputSimHits="simhits",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            # filePath=str(outputDir / "trackstates.root"),
            filePath=str(outputDir / "trackstates_gsf.root"),
        )
    )

    s.addWriter(
        RootTrackSummaryWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            # filePath=str(outputDir / "tracksummary.root"),
            filePath=str(outputDir / "tracksummary_gsf.root"),
            writeGsfSpecific=True,
        )
    )

    s.addWriter(
        RootTrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            # filePath=str(outputDir / "performance.root"),
            filePath=str(outputDir / "performance_gsf.root"),
        )
    )

    return s


if "__main__" == __name__:
    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    # ODD
    from acts.examples.odd import getOpenDataDetector

    detector = getOpenDataDetector()
    trackingGeometry = detector.trackingGeometry()
    digiConfigFile = srcdir / "Examples/Configs/odd-digi-smearing-config.json"

    ## GenericDetector
    # detector = acts.examples.GenericDetector()
    # trackingGeometry = detector.trackingGeometry()
    # digiConfigFile = (
    #     srcdir
    #     / "Examples/Configs/generic-digi-smearing-config.json"
    # )

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    print(OUTPUT_DIR)

    runTruthTrackingGsf(
        trackingGeometry=trackingGeometry,
        field=field,
        digiConfigFile=digiConfigFile,
        # outputDir=Path.cwd(),
        outputDir=OUTPUT_DIR,
        mode=MODE,
        data_type=DATA_TYPE,
        n_events=N_EVENTS,
    ).run()
