// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/PdgParticle.hpp"
#include "Acts/Definitions/Units.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Utilities/ParametricParticleGenerator.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>

namespace HepMC3 {
class GenEvent;
}

namespace ActsExamples {

/// Similar to ParametricParticleGenerator but samples theta uniformly and
/// allows sampling q/p uniformly.
class UniformThetaQOverPParticleGenerator : public ParticlesGenerator {
 public:
  struct Config {
    double phiMin = -std::numbers::pi;
    double phiMax = std::numbers::pi;

    double thetaMin = std::numeric_limits<double>::min();
    double thetaMax = std::numbers::pi - std::numeric_limits<double>::epsilon();

    // p-range if qOverPUniform is false
    double pMin = 1 * Acts::UnitConstants::GeV;
    double pMax = 10 * Acts::UnitConstants::GeV;
    // Ignored if qOverPUniform is true: q/p uniform sampling targets the
    // total momentum, not the transverse momentum.
    bool pTransverse = false;
    bool pLogUniform = false;

    // q/p uniform sampling
    bool qOverPUniform = false;
    double qOverPMin = -0.5 / Acts::UnitConstants::GeV;
    double qOverPMax = 0.5 / Acts::UnitConstants::GeV;

    Acts::PdgParticle pdg = Acts::PdgParticle::eMuon;
    bool randomizeCharge = false;
    std::size_t numParticles = 1;

    std::optional<double> charge;
    std::optional<double> mass;
  };

  explicit UniformThetaQOverPParticleGenerator(const Config& cfg);

  virtual std::shared_ptr<HepMC3::GenEvent> operator()(
      RandomEngine& rng) override;

 private:
  using UniformIndex = std::uniform_int_distribution<std::uint8_t>;
  using UniformReal = std::uniform_real_distribution<double>;

  Config m_cfg;
  double m_mass{};

  std::array<Acts::PdgParticle, 2> m_pdgChoices{};

  UniformIndex m_particleTypeChoice;
  UniformReal m_phiDist;
  std::function<std::pair<double, double>(RandomEngine& rng)> m_sinCosThetaDist;
  std::function<double(RandomEngine& rng)> m_somePDist;
  std::optional<UniformReal> m_qOverPDist;
};

}  // namespace ActsExamples
