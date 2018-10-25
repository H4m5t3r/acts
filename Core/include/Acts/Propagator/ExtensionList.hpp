// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Propagator/detail/extension_list_implementation.hpp"
#include "Acts/Utilities/detail/Extendable.hpp"
#include "Acts/Utilities/detail/MPL/all_of.hpp"
#include "Acts/Utilities/detail/MPL/has_duplicates.hpp"

namespace Acts {

template <typename... extensions>
struct ExtensionList : private detail::Extendable<extensions...>
{
private:
  static_assert(not detail::has_duplicates_v<extensions...>,
                "same extension type specified several times");

  using detail::Extendable<extensions...>::tuple;
  
  using impl = detail::extension_list_impl<extensions...>;

public:

  using detail::Extendable<extensions...>::get;

	template<typename stepper_state_t>
	bool
	k(const stepper_state_t& state,
		Vector3D&              knew,
		const Vector3D&        bField,
		const int i = 0,
	   const double           h = 0,
	   const Vector3D&        kprev = Vector3D())
    {
		return impl::k(tuple(), state, knew, bField, i, h, kprev);
	}

	template<typename stepper_state_t>
    bool
    finalize(stepper_state_t& state,const double    h,
				const Vector3D& bField1 = Vector3D(),
              const Vector3D& bField2 = Vector3D(),
              const Vector3D& bField3 = Vector3D(),
              const Vector3D& k1 = Vector3D(),
              const Vector3D& k2 = Vector3D(),
              const Vector3D& k3 = Vector3D(),
              ActsMatrixD<7, 7>& D = ActsMatrixD<7, 7>())
    {
		return impl::finalize(tuple(), state, bField1, bField2, bField3, h, k1, k2, k3, D);
	}
};

}  // namespace Acts
