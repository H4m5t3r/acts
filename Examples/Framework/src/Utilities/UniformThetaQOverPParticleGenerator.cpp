// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/Utilities/UniformThetaQOverPParticleGenerator.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/ParticleData.hpp"
#include "Acts/Utilities/AngleHelpers.hpp"
#include "Acts/Utilities/MathHelpers.hpp"

#include <limits>
#include <memory>
#include <utility>

#include <HepMC3/Attribute.h>
#include <HepMC3/FourVector.h>
#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/GenVertex.h>

using namespace Acts::UnitLiterals;

namespace ActsExamples {

UniformThetaQOverPParticleGenerator::UniformThetaQOverPParticleGenerator(
    const Config& cfg)
    : m_cfg(cfg),
      m_mass(cfg.mass.value_or(Acts::findMass(m_cfg.pdg).value_or(0))) {
  m_pdgChoices = {m_cfg.pdg, static_cast<Acts::PdgParticle>(-m_cfg.pdg)};

  m_particleTypeChoice = UniformIndex(0u, m_cfg.randomizeCharge ? 1u : 0u);
  m_phiDist = UniformReal(m_cfg.phiMin, m_cfg.phiMax);

  // sample theta uniformly (not cos(theta))
  UniformReal thetaDist(m_cfg.thetaMin, m_cfg.thetaMax);
  m_sinCosThetaDist =
      [=](RandomEngine& rng) mutable -> std::pair<double, double> {
    const double theta = thetaDist(rng);
    return {std::sin(theta), std::cos(theta)};
  };

  if (m_cfg.qOverPUniform) {
    // p (and the particle charge sign) are derived directly from the
    // sampled q/p in operator() so that the generated particle's actual
    // q/p matches the sampled value exactly; m_somePDist is unused here.
    m_qOverPDist = UniformReal(m_cfg.qOverPMin, m_cfg.qOverPMax);
  } else if (m_cfg.pLogUniform) {
    UniformReal dist(std::log(m_cfg.pMin), std::log(m_cfg.pMax));
    m_somePDist = [=](RandomEngine& rng) mutable {
      return std::exp(dist(rng));
    };
  } else {
    UniformReal dist(m_cfg.pMin, m_cfg.pMax);
    m_somePDist = [=](RandomEngine& rng) mutable { return dist(rng); };
  }
}

std::shared_ptr<HepMC3::GenEvent>
UniformThetaQOverPParticleGenerator::operator()(RandomEngine& rng) {
  auto event = std::make_shared<HepMC3::GenEvent>();

  auto primaryVertex = std::make_shared<HepMC3::GenVertex>();
  primaryVertex->set_position(HepMC3::FourVector(0., 0., 0., 0.));
  event->add_vertex(primaryVertex);

  primaryVertex->add_attribute("acts",
                               std::make_shared<HepMC3::BoolAttribute>(true));

  auto beamParticle = std::make_shared<HepMC3::GenParticle>();
  beamParticle->set_momentum(HepMC3::FourVector(0., 0., 0., 0.));
  beamParticle->set_generated_mass(0.);
  beamParticle->set_pid(Acts::PdgParticle::eInvalid);
  beamParticle->set_status(4);
  primaryVertex->add_particle_in(beamParticle);

  for (std::size_t ip = 1; ip <= m_cfg.numParticles; ++ip) {
    unsigned int type = m_particleTypeChoice(rng);
    Acts::PdgParticle pdg = m_pdgChoices[type];

    const double phi = m_phiDist(rng);

    const auto [sinTheta, cosTheta] = m_sinCosThetaDist(rng);
    const Acts::Vector3 dir = {sinTheta * std::cos(phi),
                               sinTheta * std::sin(phi), cosTheta};

    double p = 0;
    if (m_cfg.qOverPUniform && m_qOverPDist.has_value()) {
      // Derive both p and the charge sign from the same q/p draw so the
      // generated particle's actual q/p matches the sampled value exactly.
      // pTransverse does not apply here: q/p uniform sampling targets the
      // total momentum, not the transverse momentum.
      const double qop = (*m_qOverPDist)(rng);
      const double eps = std::numeric_limits<double>::min();
      const double absQop = std::abs(qop) < eps ? eps : std::abs(qop);
      p = 1.0 / absQop;
      const int absPdg = std::abs(static_cast<int>(m_cfg.pdg));
      pdg = static_cast<Acts::PdgParticle>(qop < 0 ? -absPdg : absPdg);
    } else {
      const double someP = m_somePDist(rng);
      p = someP * (m_cfg.pTransverse ? 1.0 / sinTheta : 1.0);
    }

    Acts::Vector3 momentum = p * dir;
    auto particle = std::make_shared<HepMC3::GenParticle>();
    HepMC3::FourVector hepMcMomentum(momentum.x() / 1_GeV, momentum.y() / 1_GeV,
                                     momentum.z() / 1_GeV,
                                     std::hypot(p, m_mass) / 1_GeV);
    particle->set_momentum(hepMcMomentum);
    particle->set_generated_mass(m_mass);
    particle->set_pid(pdg);
    particle->set_status(1);

    event->add_particle(particle);

    particle->add_attribute("pg_seq",
                            std::make_shared<HepMC3::UIntAttribute>(ip));
    particle->add_attribute("acts",
                            std::make_shared<HepMC3::BoolAttribute>(true));

    primaryVertex->add_particle_out(particle);
  }

  return event;
}

}  // namespace ActsExamples
