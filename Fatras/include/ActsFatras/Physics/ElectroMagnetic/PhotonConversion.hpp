// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/PdgParticle.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Utilities/UnitVectors.hpp"
#include "ActsFatras/EventData/Barcode.hpp"
#include "ActsFatras/EventData/Particle.hpp"
#include "ActsFatras/EventData/ProcessType.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

namespace ActsFatras {

/// This class handles the photon conversion. It evaluates the distance
/// after which the interaction will occur and the final state due the
/// interaction itself.
class PhotonConversion {
 public:
  /// Scaling factor of children energy
  double childEnergyScaleFactor = 2.;
  /// Scaling factor for photon conversion probability
  double conversionProbScaleFactor = 0.98;

  /// Method for evaluating the distance after which the photon
  /// conversion will occur.
  ///
  /// @tparam generator_t Type of the random number generator
  /// @param [in, out] generator The random number generator
  /// @param [in] particle The particle
  ///
  /// @return valid X0 limit and no limit on L0
  template <typename generator_t>
  std::pair<double, double> generatePathLimits(generator_t& generator,
                                               const Particle& particle) const;

  /// This method evaluates the final state due to the photon conversion.
  ///
  /// @tparam generator_t Type of the random number generator
  /// @param [in, out] generator The random number generator
  /// @param [in, out] particle The interacting photon
  /// @param [out] generated List of generated particles
  ///
  /// @return True if the conversion occurred, else false
  template <typename generator_t>
  bool run(generator_t& generator, Particle& particle,
           std::vector<Particle>& generated) const;

 private:
  /// This method constructs and returns the child particles.
  ///
  /// @param [in] photon The interacting photon
  /// @param [in] childEnergy The energy of one child particle
  /// @param [in] childDirection The direction of the child particle
  ///
  /// @return Array containing the produced leptons
  std::array<Particle, 2> generateChildren(
      const Particle& photon, double childEnergy,
      const Acts::Vector3& childDirection) const;

  /// Generate the energy fraction of the first child particle.
  ///
  /// @tparam generator_t Type of the random number generator
  /// @param [in, out] generator The random number generator
  /// @param [in] gammaMom The momentum of the photon
  ///
  /// @return The energy of the child particle
  template <typename generator_t>
  double generateFirstChildEnergyFraction(generator_t& generator,
                                          double gammaMom) const;

  /// Generate the direction of the child particles.
  ///
  /// @tparam generator_t Type of the random number generator
  /// @param [in, out] generator The random number generator
  /// @param [in] particle The photon
  ///
  /// @return The direction vector of the child particle
  template <typename generator_t>
  Acts::Vector3 generateChildDirection(generator_t& generator,
                                       const Particle& particle) const;

  /// Helper methods for momentum evaluation
  /// @note These methods are taken from the Geant4 class
  /// G4PairProductionRelModel
  double screenFunction1(double delta) const;
  double screenFunction2(double delta) const;

  /// Electron mass. This is an static constant and not a member variable so the
  /// struct has no internal state. Otherwise, the interaction list breaks.
  static const double kElectronMass;
};

inline double PhotonConversion::screenFunction1(double delta) const {
  // Compute the value of the screening function 3*PHI1(delta) - PHI2(delta)
  return (delta > 1.4) ? 42.038 - 8.29 * std::log(delta + 0.958)
                       : 42.184 - delta * (7.444 - 1.623 * delta);
}

inline double PhotonConversion::screenFunction2(double delta) const {
  // Compute the value of the screening function 1.5*PHI1(delta)
  // +0.5*PHI2(delta)
  return (delta > 1.4) ? 42.038 - 8.29 * std::log(delta + 0.958)
                       : 41.326 - delta * (5.848 - 0.902 * delta);
}

template <typename generator_t>
std::pair<double, double> PhotonConversion::generatePathLimits(
    generator_t& generator, const Particle& particle) const {
  /// This method is based upon the Athena class PhotonConversionTool

  // Fast exit if not a photon or the energy is too low
  if (particle.pdg() != Acts::PdgParticle::eGamma ||
      particle.absoluteMomentum() < (2 * kElectronMass)) {
    return {std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
  }

  // Use for the moment only Al data - Yung Tsai - Rev.Mod.Particle Physics Vol.
  // 46, No.4, October 1974 optainef from a fit given in the momentum range 100
  // 10 6 2 1 0.6 0.4 0.2 0.1 GeV

  //// Quadratic background function
  //  Double_t fitFunction(Double_t *x, Double_t *par) {
  //  return par[0] + par[1]*pow(x[0],par[2]);
  // }
  // EXT PARAMETER                                   STEP         FIRST
  // NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE
  //  1  p0          -7.01612e-03   8.43478e-01   1.62766e-04   1.11914e-05
  //  2  p1           7.69040e-02   1.00059e+00   8.90718e-05  -8.41167e-07
  //  3  p2          -6.07682e-01   5.13256e+00   6.07228e-04  -9.44448e-07
  constexpr double p0 = -7.01612e-03;
  constexpr double p1 = 7.69040e-02;
  constexpr double p2 = -6.07682e-01;

  // Calculate xi
  const double xi = p0 + p1 * std::pow(particle.absoluteMomentum(), p2);

  std::uniform_real_distribution<double> uniformDistribution{0., 1.};
  // This is a transformation of eq. 3.75
  return {-9. / 7. *
              std::log(conversionProbScaleFactor *
                       (1 - uniformDistribution(generator))) /
              (1. - xi),
          std::numeric_limits<double>::infinity()};
}

template <typename generator_t>
double PhotonConversion::generateFirstChildEnergyFraction(
    generator_t& generator, double gammaMom) const {
  /// This method is based upon the Geant4 class G4PairProductionRelModel

  /// @note This method is from the Geant4 class G4Element
  //
  //  Compute Coulomb correction factor (Phys Rev. D50 3-1 (1994) page 1254)
  constexpr double k1 = 0.0083;
  constexpr double k2 = 0.20206;
  constexpr double k3 = 0.0020;  // This term is missing in Athena
  constexpr double k4 = 0.0369;
  constexpr double alphaEM = 1. / 137.;
  constexpr double m_Z = 13.;  // Aluminium
  constexpr double az2 = (alphaEM * m_Z) * (alphaEM * m_Z);
  constexpr double az4 = az2 * az2;
  constexpr double coulombFactor =
      (k1 * az4 + k2 + 1. / (1. + az2)) * az2 - (k3 * az4 + k4) * az4;

  const double logZ13 = std::log(m_Z) * 1. / 3.;
  const double FZ = 8. * (logZ13 + coulombFactor);
  const double deltaMax = std::exp((42.038 - FZ) * 0.1206) - 0.958;

  const double deltaPreFactor = 136. / std::pow(m_Z, 1. / 3.);
  const double eps0 = kElectronMass / gammaMom;
  const double deltaFactor = deltaPreFactor * eps0;
  const double deltaMin = 4. * deltaFactor;

  // Compute the limits of eps
  const double epsMin =
      std::max(eps0, 0.5 - 0.5 * std::sqrt(1. - deltaMin / deltaMax));
  const double epsRange = 0.5 - epsMin;

  // Sample the energy rate (eps) of the created electron (or positron)
  const double F10 = screenFunction1(deltaMin) - FZ;
  const double F20 = screenFunction2(deltaMin) - FZ;
  const double NormF1 = F10 * epsRange * epsRange;
  const double NormF2 = 1.5 * F20;

  // We will need 3 uniform random number for each trial of sampling
  double greject = 0.;
  double eps = 0.;
  std::uniform_real_distribution<double> rndmEngine;
  do {
    if (NormF1 > rndmEngine(generator) * (NormF1 + NormF2)) {
      eps = 0.5 - epsRange * std::pow(rndmEngine(generator), 1. / 3.);
      const double delta = deltaFactor / (eps * (1. - eps));
      greject = (screenFunction1(delta) - FZ) / F10;
    } else {
      eps = epsMin + epsRange * rndmEngine(generator);
      const double delta = deltaFactor / (eps * (1. - eps));
      greject = (screenFunction2(delta) - FZ) / F20;
    }
  } while (greject < rndmEngine(generator));
  //  End of eps sampling
  return eps * childEnergyScaleFactor;
}

template <typename generator_t>
Acts::Vector3 PhotonConversion::generateChildDirection(
    generator_t& generator, const Particle& particle) const {
  /// This method is based upon the Athena class PhotonConversionTool

  // Following the Geant4 approximation from L. Urban
  // the azimutal angle
  double theta = kElectronMass / particle.energy();

  std::uniform_real_distribution<double> uniformDistribution{0., 1.};
  const double u = -std::log(uniformDistribution(generator) *
                             uniformDistribution(generator)) *
                   1.6;

  theta *= (uniformDistribution(generator) < 0.25)
               ? u
               : u * 1. / 3.;  // 9./(9.+27) = 0.25

  // draw the random orientation angle
  const auto psi = std::uniform_real_distribution<double>(
      -std::numbers::pi, std::numbers::pi)(generator);

  Acts::Vector3 direction = particle.direction();
  // construct the combined rotation to the scattered direction
  Acts::RotationMatrix3 rotation(
      // rotation of the scattering deflector axis relative to the reference
      Acts::AngleAxis3(psi, direction) *
      // rotation by the scattering angle around the deflector axis
      Acts::AngleAxis3(theta, Acts::createCurvilinearUnitU(direction)));
  direction.applyOnTheLeft(rotation);
  return direction;
}

inline std::array<Particle, 2> PhotonConversion::generateChildren(
    const Particle& photon, double childEnergy,
    const Acts::Vector3& childDirection) const {
  using namespace Acts::UnitLiterals;

  // Calculate the child momentum
  const double massChild = kElectronMass;
  const double momentum1 =
      sqrt(childEnergy * childEnergy - massChild * massChild);

  // Use energy-momentum conservation for the other child
  const Acts::Vector3 vtmp =
      photon.fourMomentum().template segment<3>(Acts::eMom0) -
      momentum1 * childDirection;
  const double momentum2 = vtmp.norm();

  // The daughter particles are created with the explicit electron mass used in
  // the calculations for consistency. Using the full Particle constructor with
  // charge and mass also avoids an additional lookup in the internal data
  // tables.
  std::array<Particle, 2> children = {
      Particle(photon.particleId().makeDescendant(0), Acts::eElectron, -1_e,
               kElectronMass)
          .setPosition4(photon.fourPosition())
          .setDirection(childDirection)
          .setAbsoluteMomentum(momentum1)
          .setProcess(ProcessType::ePhotonConversion)
          .setReferenceSurface(photon.referenceSurface()),
      Particle(photon.particleId().makeDescendant(1), Acts::ePositron, 1_e,
               kElectronMass)
          .setPosition4(photon.fourPosition())
          .setDirection(childDirection)
          .setAbsoluteMomentum(momentum2)
          .setProcess(ProcessType::ePhotonConversion)
          .setReferenceSurface(photon.referenceSurface()),
  };
  return children;
}

template <typename generator_t>
bool PhotonConversion::run(generator_t& generator, Particle& particle,
                           std::vector<Particle>& generated) const {
  // Fast exit if particle is not a photon
  if (particle.pdg() != Acts::PdgParticle::eGamma) {
    return false;
  }

  // Fast exit if momentum is too low
  const double p = particle.absoluteMomentum();
  if (p < (2 * kElectronMass)) {
    return false;
  }

  // Get one child energy
  const double childEnergy = p * generateFirstChildEnergyFraction(generator, p);

  // Now get the deflection
  const Acts::Vector3 childDir = generateChildDirection(generator, particle);

  // Produce the final state
  const std::array<Particle, 2> finalState =
      generateChildren(particle, childEnergy, childDir);
  generated.insert(generated.end(), finalState.begin(), finalState.end());

  return true;
}

}  // namespace ActsFatras
