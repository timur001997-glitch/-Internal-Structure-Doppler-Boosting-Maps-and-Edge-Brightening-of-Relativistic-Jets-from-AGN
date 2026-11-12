# -Internal-Structure-Doppler-Boosting-Maps-and-Edge-Brightening-of-Relativistic-Jets-from-AGN
This repository contains calculations of the local magnetization, Lorentz factor, and Doppler boosting in the synchrotron radiation regions of relativistic jets from active galactic nuclei within the framework of the Grad-Shafranov method (stationary axisymmetric magnetohydrodynamics).

The following calculations are presented here:
"Lorentz_factors_for_equations.py" shows the comparison of Lorentz factors calculated using different methods for the real profile of relativistic jet with the dense central core;

"Schematic_figure_for_parabolic_model_with_B_p=const.py" presents the schematic illustration of the boundaries of regions where the magnetic flux dominates over the particle flux in the relativistic jet with the uniform poloidal magnetic field;

"Transverse_profiles_of_Lorentz_factors_for_parabolic_model_with_B_p=const.py" calculates the dependencies the Lorentz factor and Doppler factor for the uniform poloidal magnetic field for small and large distances from the central engine;

"Map_of_Doppler_factor_for_parabolic_model_with_B_p=const.py" calculates the distributions of the local magnetization (the ratio of the electromagnetic energy flux to the particle energy flux), the hydrodynamic Lorentz factor, the Doppler factor, and Poynting flux for emitters in the yz-plane in the laboratory reference frame for the case of the uniform poloidal magnetic field;

"Map_of_Doppler_factor_for_parabolic_model_with_B_p=const_for_new_v_phi.py" calculates similarly "Map_of_Doppler_factor_for_parabolic_model_with_B_p=const.py", but for the different approximation of the toroidal velocity inside the light cylinder;

"Map_of_Doppler_factor_for_parabolic_model_with_central_core.py" calculates the distributions of the local magnetization (the ratio of the electromagnetic energy flux to the particle energy flux), the hydrodynamic Lorentz factor, the Doppler factor, and Poynting flux for emitters in the yz-plane in the laboratory reference frame for the case of the nonuniform poloidal magnetic field with the dense central core;

"Map_of_Doppler_factor_for_parabolic_model_with_central_core_for_new_v_phi.py" calculates similarly "Map_of_Doppler_factor_for_parabolic_model_with_central_core.py", but for the different approximation of the toroidal velocity inside the light cylinder;

"Test_T_br.py" simulates radiation transfer and builds brightness temperature maps for the model with the homogeneous poloidal magnetic field (Preliminary version);

"Test_T_br_for_new_v_phi.py" calculates similarly "Test_T_br.py", but for the different approximation of the toroidal velocity inside the light cylinder;

"Test_T_br._with_core.py" simulates radiation transfer and builds brightness temperature maps for the model with the dense central core in the poloidal magnetic field (Preliminary version);

"Test_T_br_with_core_for_new_v_phi.py" calculates similarly "Test_T_br._with_core.py", but for the different approximation of the toroidal velocity inside the light cylinder.

This work was supported by Russian Science Foundation grant 26-12-00190.

If these numerical calculations are useful for your research, please cite it: https://doi.org/10.5281/zenodo.23974669, or directly through this repository on GitHub.
