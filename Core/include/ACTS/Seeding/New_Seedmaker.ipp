#include <cmath>
#include "ACTS/Seeding/ICovarianceTool.hpp"
#include "ACTS/Seeding/IBinFinder.hpp"
#include "ACTS/Seeding/ISeedFilter.hpp"


// TODO: remove all debug output
#include <iostream>


namespace Acts{
namespace Seeding{

New_Seedmaker::New_Seedmaker(const Acts::Seeding::Config& config): m_config(config){

}

std::shared_ptr<Acts::Seeding::Cache>
New_Seedmaker::initialize() const
{
  
  auto cache = std::make_shared<Acts::Seeding::Cache>();
  // back-of-the-envelope calculation of scattering, leaving out the insignificant term
  // of the highland formula
  // convert pT to p once theta angle is known
  cache->highland =  13.6*sqrt(m_config.radLengthPerSeed)*(1+0.038*log(m_config.radLengthPerSeed));
  float maxScatteringAngle = cache->highland/m_config.minPt;
  cache->maxScatteringAngle2 = maxScatteringAngle*maxScatteringAngle;
  // helix radius in homogeneous magnetic field. units are Tesla, MeV and millimeter
  // TODO: change using ACTS units
  cache->pTPerHelixRadius = 300.*m_config.bFieldInZ;
  cache->minHelixRadius2 = std::pow(m_config.minPt/cache->pTPerHelixRadius,2);
  return cache;
}

void 
New_Seedmaker::newEvent
( std::vector<const Acts::concept::AnySpacePoint<>*> spVec, 
  std::shared_ptr<Acts::Seeding::Cache> cache) const
{
// TODO: clear everything!
  cache->seeds.clear();
  std::unique_ptr<SPGrid> grid = SPGridCreator::createGrid(m_config);
  float phiMin = m_config.minPhi;
  float phiMax = m_config.maxPhi;
  float zMin = m_config.zMin;
  float zMax = m_config.zMax;

  std::sort(spVec.begin(),spVec.end(),comR());
  for(auto sp : spVec){
    std::array<float,2> cov = m_config.covarianceTool->getCovariances(sp,m_config.zAlign, m_config.rAlign, m_config.sigmaError);
    SPForSeed sps(sp, m_config.beamPos,cov);
    float spPhi = sps.phi();
    if(spPhi > phiMax || spPhi < phiMin) continue;
    float spZ = sps.z();
    if(spZ > zMax || spZ < zMin) continue;
    Acts::Vector2D spLocation(spPhi,spZ);
    std::vector<std::shared_ptr<SPForSeed > >& bin = grid->at(spLocation);
    bin.push_back(std::make_shared<SPForSeed>(sps));
  }
  cache->binnedSP = std::move(grid);
}


std::vector<std::shared_ptr<Seed> > 
New_Seedmaker::production3Sp
( std::shared_ptr<Acts::Seeding::Cache> cache) const
{
  std::vector<std::shared_ptr<Seed> > outputSeeds;
  // TODO: create neighborHoodIndices in BinFinder so it can be replaced by smarter neighbor choice
  auto phiZbins = cache->binnedSP->getNBins();
  for (size_t i =1; i <= phiZbins[0]; ++i){
    for (size_t j =1; j <= phiZbins[1]; ++j){
    // if different combinations of spacepoints (i.e. only pixel, pixel + sct, only sct) should be 
    // treated differently, call multiple times with different config and with findBottomBins (findTopBins)
    // returning the corresponding space points
      std::set<size_t > bottomBins = m_config.bottomBinFinder->findBins(i,j,cache->binnedSP);
      std::set<size_t > topBins = m_config.topBinFinder->findBins(i,j,cache->binnedSP);
      std::vector<std::shared_ptr<Seed> > regionSeeds = production3Sp(cache->binnedSP->at({i,j}), bottomBins, topBins, cache);
      outputSeeds.insert(outputSeeds.end(),regionSeeds.begin(),regionSeeds.end());
    }
  }
  return outputSeeds;
}


std::vector<std::shared_ptr<Seed> > 
New_Seedmaker::production3Sp
( std::vector<std::shared_ptr<SPForSeed > > currentBin,
  std::set<size_t > bottomBinIndices,
  std::set<size_t > topBinIndices,
  std::shared_ptr<Acts::Seeding::Cache> cache) const
{
  std::vector<std::shared_ptr<SPForSeed> > compatBottomSP, compatTopSP;
  std::vector<std::shared_ptr<InternalSeed> > regionSeeds;

  // middle space point
  for(auto spM : currentBin){
    float rM = spM->radius();
    float zM = spM->z();
    float covrM = spM->covr();
    float covzM = spM->covz();

    compatBottomSP.clear();

    // bottom space point
    for(auto bottomBinIndex : bottomBinIndices){
      auto bottomBin = cache->binnedSP->at(bottomBinIndex);
      for(auto spB : bottomBin){
        float rB = spB->radius();
        float deltaR = rM - rB;
        // if r-distance is too big, try next SP in r-sorted bin
        if (deltaR > m_config.deltaRMax) continue;
        // if r-distance is too small, break because bins are r-sorted
        if (deltaR < m_config.deltaRMin) break;
        // ratio Z/R (forward angle) of space point duplet
        float cotTheta = (zM-spB->z())/deltaR;
        if(std::fabs(cotTheta) > m_config.cotThetaMax) continue;
        // check if duplet origin on z axis within collision region
        float zOrigin = zM-rM*cotTheta;
        if(zOrigin < m_config.collisionRegionMin || zOrigin > m_config.collisionRegionMax) continue;
        compatBottomSP.push_back(spB);
      }
    }
    // no bottom SP found -> try next spM
    if(compatBottomSP.size()==0) continue;
    
    compatTopSP.clear();

    for(auto topBinIndex : topBinIndices){ 
      auto topBin = cache->binnedSP->at(topBinIndex);
      for (auto spT : topBin){
        float rT = spT->radius();
        float deltaR = rT-rM;
        // this condition is the opposite of the condition for bottom SP
        if(deltaR < m_config.deltaRMin ) continue;
        if(deltaR > m_config.deltaRMax ) break;

        float cotTheta = (spT->z()-zM)/deltaR;
        if(std::fabs(cotTheta) > m_config.cotThetaMax) continue;
        float zOrigin = zM-rM*cotTheta;
        if(zOrigin < m_config.collisionRegionMin || zOrigin > m_config.collisionRegionMax) continue;
        compatTopSP.push_back(spT);
      }
    }
    if(compatTopSP.size()==0) continue;
    cache->linCircleBottom.clear();
    transformCoordinates(compatBottomSP, spM, true, cache->linCircleBottom);
    cache->linCircleTop.clear();
    transformCoordinates(compatTopSP, spM, false, cache->linCircleTop);
    
    // TODO: significant benefit? avoids compatSp.size()^2 reallocations
    // create vectors here to avoid reallocation in each loop
    std::vector<std::shared_ptr<SPForSeed> > topSpVec;
    std::vector<float > curvatures, impactParameters;

    // TODO: measure cost to reallocate seedsPerSpM each iteration
    std::vector<std::pair<float,std::shared_ptr<InternalSeed > > > seedsPerSpM;

    for(size_t b = 0; b < compatBottomSP.size(); b++){
      auto lb = cache->linCircleBottom.at(b);
      float  Zob  = lb.Zo      ;
      float  cotThetaB = lb.cotTheta ;
      float  Vb   = lb.V       ;
      float  Ub   = lb.U       ;
      float  ErB   = lb.Er     ;
      float  iDeltaRB = lb.iDeltaR;

      // 1+(cot^2(theta)) = 1/sin^2(theta)
      float iSinTheta2 = (1.+cotThetaB*cotThetaB) ;
      // calculate max scattering for min momentum at the seed's theta angle
      // scaling scatteringAngle^2 by sin^2(theta) to convert pT^2 to p^2
      // accurate would be taking 1/atan(thetaBottom)-1/atan(thetaTop) < scattering
      // but to avoid trig functions we approximate cot by scaling by 1/sin^4(theta)
      // resolving with pT to p scaling --> only divide by sin^2(theta)
      // max approximation error for allowed scattering angles of 0.04 rad at eta=0: ~8.5%
      float scatteringInRegion2 = cache->maxScatteringAngle2 * iSinTheta2;
      std::cout << "max scattering angle in this detector region (1 sigma): " << sqrt(scatteringInRegion2) << std::endl;
      // multiply the squared sigma onto the squared scattering
      scatteringInRegion2 *= m_config.sigmaScattering*m_config.sigmaScattering;

      // clear all vectors used in each inner for loop
      topSpVec.clear();
      curvatures.clear();
      impactParameters.clear();
      for(size_t t = 0; t < compatTopSP.size(); t++) {
        auto lt = cache->linCircleTop.at(t);

        // add errors of spB-spM and spM-spT pairs and add the correlation term for errors on spM
        float error = lt.Er + ErB + 2*(cotThetaB * lt.cotTheta * covrM + covzM) * iDeltaRB * lt.iDeltaR;

        float deltaCotTheta = cotThetaB - lt.cotTheta;
        float dCotThetaCorrected = deltaCotTheta*deltaCotTheta - error;

        // if deltaTheta larger than the scattering for the lower pT cut, skip
        if ( dCotThetaCorrected > scatteringInRegion2) continue;

        // protects against division by 0
        float dU  = lt.U-Ub; if(dU == 0.) continue ;
        // A and B are evaluated as a function of the circumference parameters x_0 and y_0
        float A   = (lt.V-Vb)/dU                     ;
        float S2  = 1.+A*A                           ;
        float B   = Vb-A*Ub                          ;
        float B2  = B*B                              ;
        // sqrt(S2)/B = 2 * helixradius
        // calculated radius must not be smaller than minimum radius
        if(S2 < B2*cache->minHelixRadius2*2) continue;
        // 1/helixradius: (B/sqrt(S2))/2 (we leave everything squared)
        float iHelixradius2 = 4*B2/S2;
        // calculate scattering for p(T) calculated from seed curvature
        float pT2perRadius = cache->highland/cache->pTPerHelixRadius;
        pT2perRadius = pT2perRadius*pT2perRadius;
        float pT2scatter = iHelixradius2 * pT2perRadius;
        pT2scatter *= pT2scatter;
        //convert p(T) to p scaling by sin^2(theta) AND scale by 1/sin^4(theta) from rad to deltaCotTheta
        float p2scatter = pT2scatter * iSinTheta2;
        // if deltaTheta larger than allowed scattering for calculated pT, skip
        if(dCotThetaCorrected > p2scatter * m_config.sigmaScattering* m_config.sigmaScattering) continue;
        // A and B allow calculation of impact params in U/V plane with linear function
        // (in contrast to x^2 in x/y plane)
        float Im  = fabs((A-B*rM)*rM)                ;

        if(Im <= m_config.impactMax) {
          // TODO: test impact of at() instead of [] access. should never be out of bounds.
          topSpVec.push_back(compatTopSP.at(t));
          curvatures.push_back(sqrt(iHelixradius2));
          impactParameters.push_back(Im);
        }
      }
      if(!topSpVec.empty()) {
        std::vector<std::pair<float, std::shared_ptr<InternalSeed > > > sameTrackSeeds;
        sameTrackSeeds = m_config.seedFilter->filterSeeds_2SpFixed(compatBottomSP.at(b),
                                                               spM,
                                                               topSpVec,
                                                               curvatures,
                                                               impactParameters,
                                                               Zob);
        seedsPerSpM.insert(seedsPerSpM.end(), sameTrackSeeds.begin(), sameTrackSeeds.end());
      }
    }
    std::vector<std::shared_ptr<InternalSeed> > filteredSpMSeeds;
    filteredSpMSeeds = m_config.seedFilter->filterSeeds_1SpFixed(seedsPerSpM);
    regionSeeds.insert(regionSeeds.end(), filteredSpMSeeds.begin(), filteredSpMSeeds.end());
  }
  return m_config.seedFilter->filterSeeds_byRegion(regionSeeds);
}
  

void New_Seedmaker::transformCoordinates
( std::vector<std::shared_ptr<SPForSeed> >& vec,
  std::shared_ptr<SPForSeed> spM,
  bool bottom,
  std::vector<LinCircle>& linCircleVec) const
{
  float xM = spM->x();
  float yM = spM->y();
  float zM = spM->z();
  float rM = spM->radius();
  float covzM = spM->covz();
  float covrM = spM->covr();
  float cosPhiM = xM/rM;
  float sinPhiM = yM/rM;
  for (auto sp : vec){
    float deltaX = sp->x()-xM;
    float deltaY = sp->y()-yM;
    float deltaZ = sp->z()-zM;
    // calculate projection fraction of spM->sp vector pointing in same direction as
    // vector origin->spM (x) and projection fraction of spM->sp vector pointing 
    // orthogonal to origin->spM (y)
    float x = deltaX * cosPhiM + deltaY*sinPhiM;
    float y = deltaY * cosPhiM - deltaX*sinPhiM;
    // 1/(deltaR*deltaR)
    // x*x+y*y is larger for smaller impact params and higher pT
    // x*x+y*y is always > 1 (unless pT is too low for this Seedfinder)
    // 1/(length of M -> SP) 
    float iDeltaR2 = 1./(deltaX*deltaX+deltaY*deltaY);
    float iDeltaR = sqrt(iDeltaR2);
    // 
    int bottomFactor = 1 * (!bottom) - 1*bottom ;
    // cot_theta = (deltaZ/deltaR)
    float cot_theta = deltaZ*iDeltaR*bottomFactor;
    // VERY frequent (SP^3) access
    LinCircle l;
    l.cotTheta   = cot_theta                                        ;
    // location on z-axis of this SP-duplet
    l.Zo         = zM-rM * cot_theta                                ;
    l.iDeltaR    = iDeltaR                                          ;
    // transformation of circle equation (x,y) into linear equation (u,v)
    // x^2 + y^2 - 2x_0*x - 2y_0*y = 0
    // is transformed into
    // 1 - 2x_0*u - 2y_0*v = 0
    // using the following m_U and m_V
    // (u = A + B*v); A and B are created later on
    l.U    = x*iDeltaR2                                             ;
    l.V    = y*iDeltaR2                                             ;
    //error term for sp-pair without correlation of middle space point
    l.Er   = ((covzM+sp->covz())+(cot_theta*cot_theta)*(covrM+sp->covr()))*iDeltaR2;
    linCircleVec.push_back(l);
  }
}
} // Seeding namespace
} // Acts namespace
