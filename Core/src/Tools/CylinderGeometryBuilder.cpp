// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// CylinderGeometryBuilder.cpp, ACTS project
///////////////////////////////////////////////////////////////////

#include "ACTS/Tools/CylinderGeometryBuilder.hpp"
#include "ACTS/Detector/TrackingGeometry.hpp"
#include "ACTS/Detector/TrackingVolume.hpp"
#include "ACTS/Tools/ITrackingVolumeBuilder.hpp"
#include "ACTS/Tools/ITrackingVolumeHelper.hpp"
#include "ACTS/Volumes/CylinderVolumeBounds.hpp"

Acts::CylinderGeometryBuilder::CylinderGeometryBuilder(
    const Acts::CylinderGeometryBuilder::Config& cgbConfig,
    std::unique_ptr<Logger>                      logger)
  : m_cfg(), m_logger(std::move(logger))
{
  setConfiguration(cgbConfig);
}

void
Acts::CylinderGeometryBuilder::setConfiguration(
    const Acts::CylinderGeometryBuilder::Config& cgbConfig)
{
  // @TODO check consistency
  // copy the configuration
  m_cfg = cgbConfig;
}

void
Acts::CylinderGeometryBuilder::setLogger(std::unique_ptr<Logger> newLogger)
{
  m_logger = std::move(newLogger);
}

std::unique_ptr<Acts::TrackingGeometry>
Acts::CylinderGeometryBuilder::trackingGeometry() const
{
  // the return geometry -- and the highest volume
  std::unique_ptr<Acts::TrackingGeometry> trackingGeometry;
  TrackingVolumePtr                       highestVolume = nullptr;
  // loop over the builders and wrap one around the other
  // -----------------------------
  for (auto& volumeBuilder : m_cfg.trackingVolumeBuilders) {
    // assign a new highest volume (and potentially wrap around the given
    // highest volume so far)
    highestVolume = volumeBuilder->trackingVolume(highestVolume);
  }  // --------------------------------------------------------------------------------
  // if you have a highst volume, stuff it into a TrackingGeometry
  if (highestVolume) {
    // see if the beampipe needs to be wrapped
    if (m_cfg.beamPipeBuilder && m_cfg.trackingVolumeHelper) {
      // some screen output
      ACTS_DEBUG("BeamPipe is being built and inserted.");
      // cast to cylinder volume bounds
      const CylinderVolumeBounds* cvB
          = dynamic_cast<const CylinderVolumeBounds*>(
              &(highestVolume->volumeBounds()));
      if (cvB) {
        // get the inner radius
        double innerR = cvB->innerRadius();
        double halfZ  = cvB->halflengthZ();
        // create bounds for the innermost Volume
        VolumeBoundsPtr beamPipeBounds(
            new CylinderVolumeBounds(0., innerR, halfZ));
        TrackingVolumePtr beamPipeVolume
            = m_cfg.beamPipeBuilder->trackingVolume(nullptr, beamPipeBounds);
        // update the highest volume with the beam pipe
        highestVolume
            = m_cfg.trackingVolumeHelper->createContainerTrackingVolume(
                {beamPipeVolume, highestVolume});
      }
    }
    // create the TrackingGeoemtry
    trackingGeometry.reset(new Acts::TrackingGeometry(highestVolume));
  }
  // return the geometry to the service
  return (trackingGeometry);
}
