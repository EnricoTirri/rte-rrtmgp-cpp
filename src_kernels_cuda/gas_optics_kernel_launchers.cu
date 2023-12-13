#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>

#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"
#include <omp.h>


namespace {

#include "gas_optics_kernels.cu"

}


namespace rrtmgp_kernel_launcher_cuda {
    void reorder123x321(
            const int ni, const int nj, const int nk,
            const Float *arr_in, Float *arr_out) {
        Tuner_map &tunings = Tuner::get_map();

        dim3 grid(ni, nj, nk);
        dim3 block;

        if (tunings.count("reorder123x321_kernel") == 0) {
            std::tie(grid, block) = tune_kernel(
                    "reorder123x321_kernel",
                    dim3(ni, nj, nk),
                    {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                    {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                    {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                    reorder123x321_kernel,
                    ni, nj, nk, arr_in, arr_out);

            tunings["reorder123x321_kernel"].first = grid;
            tunings["reorder123x321_kernel"].second = block;
        } else {
            grid = tunings["reorder123x321_kernel"].first;
            block = tunings["reorder123x321_kernel"].second;
        }

        reorder123x321_kernel<<<grid, block>>>(
                ni, nj, nk, arr_in, arr_out);
    }


    void reorder12x21(
            const int ni, const int nj,
            const Float *arr_in, Float *arr_out) {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i = ni / block_i + (ni % block_i > 0);
        const int grid_j = nj / block_j + (nj % block_j > 0);

        dim3 grid_gpu(grid_i, grid_j);
        dim3 block_gpu(block_i, block_j);

        reorder12x21_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr_in, arr_out);
    }


    void zero_array(const int ni, const int nj, const int nk, Float *arr) {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i = ni / block_i + (ni % block_i > 0);
        const int grid_j = nj / block_j + (nj % block_j > 0);
        const int grid_k = nk / block_k + (nk % block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr);

    }


    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int *flavor,
            const Float *press_ref_log,
            const Float *temp_ref,
            Float press_ref_log_delta,
            Float temp_ref_min,
            Float temp_ref_delta,
            Float press_ref_trop_log,
            const Float *vmr_ref,
            const Float *play,
            const Float *tlay,
            Float *col_gas,
            int *jtemp,
            Float *fmajor, Float *fminor,
            Float *col_mix,
            Bool *tropo,
            int *jeta,
            int *jpress) {
        const int block_col = 4;
        const int block_lay = 2;
        const int block_flav = 16;

        const int grid_col = ncol / block_col + (ncol % block_col > 0);
        const int grid_lay = nlay / block_lay + (nlay % block_lay > 0);
        const int grid_flav = nflav / block_flav + (nflav % block_flav > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_flav);
        dim3 block_gpu(block_col, block_lay, block_flav);

        Float tmin = std::numeric_limits<Float>::min();
        interpolation_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                flavor, press_ref_log, temp_ref,
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref, play, tlay,
                col_gas, jtemp, fmajor,
                fminor, col_mix, tropo,
                jeta, jpress);
    }


    void combine_abs_and_rayleigh(
            const int ncol, const int nlay, const int ngpt,
            const Float *tau_abs, const Float *tau_rayleigh,
            Float *tau, Float *ssa, Float *g) {
        Tuner_map &tunings = Tuner::get_map();

        Float tmin = std::numeric_limits<Float>::min();

        dim3 grid(ncol, nlay, ngpt);
        dim3 block;

        if (tunings.count("combine_abs_and_rayleigh_kernel") == 0) {
            std::tie(grid, block) = tune_kernel(
                    "combine_abs_and_rayleigh_kernel",
                    dim3(ncol, nlay, ngpt),
                    {1, 2, 4, 8, 16, 24, 32, 48, 64, 96}, {1, 2, 4}, {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                    combine_abs_and_rayleigh_kernel,
                    ncol, nlay, ngpt, tmin,
                    tau_abs, tau_rayleigh,
                    tau, ssa, g);

            tunings["combine_abs_and_rayleigh_kernel"].first = grid;
            tunings["combine_abs_and_rayleigh_kernel"].second = block;
        } else {
            grid = tunings["combine_abs_and_rayleigh_kernel"].first;
            block = tunings["combine_abs_and_rayleigh_kernel"].second;
        }

        combine_abs_and_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);
    }


    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int *gpoint_flavor,
            const int *gpoint_bands,
            const int *band_lims_gpt,
            const Float *krayl,
            int idx_h2o, const Float *col_dry, const Float *col_gas,
            const Float *fminor, const int *jeta,
            const Bool *tropo, const int *jtemp,
            Float *tau_rayleigh) {
        Tuner_map &tunings = Tuner::get_map();

        dim3 grid(ncol, nlay, ngpt);
        dim3 block;

        if (tunings.count("compute_tau_rayleigh_kernel") == 0) {
            std::tie(grid, block) = tune_kernel(
                    "compute_tau_rayleigh_kernel",
                    dim3(ncol, nlay, ngpt),
                    {1, 2, 4, 16, 24, 32}, {1, 2, 4}, {1, 2, 4, 8, 16},
                    compute_tau_rayleigh_kernel,
                    ncol, nlay, nbnd, ngpt,
                    ngas, nflav, neta, npres, ntemp,
                    gpoint_flavor,
                    gpoint_bands,
                    band_lims_gpt,
                    krayl,
                    idx_h2o, col_dry, col_gas,
                    fminor, jeta,
                    tropo, jtemp,
                    tau_rayleigh);

            tunings["compute_tau_rayleigh_kernel"].first = grid;
            tunings["compute_tau_rayleigh_kernel"].second = block;
        } else {
            grid = tunings["compute_tau_rayleigh_kernel"].first;
            block = tunings["compute_tau_rayleigh_kernel"].second;
        }

        compute_tau_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor,
                gpoint_bands,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);
    }


    struct Gas_optical_depths_major_kernel {
        template<unsigned int I, unsigned int J, unsigned int K, class... Args>
        static void launch(dim3 grid, dim3 block, Args... args) {
            gas_optical_depths_major_kernel<I, J, K><<<grid, block>>>(args...);
        }
    };


    struct Gas_optical_depths_minor_kernel {
        template<unsigned int I, unsigned int J, unsigned int K, class... Args>
        static void launch(dim3 grid, dim3 block, Args... args) {
            gas_optical_depths_minor_kernel<I, J, K><<<grid, block>>>(args...);
        }
    };

    template<int block_size_y, int block_size_z, int n_block_y, int n_block_z>
    void serial_kernel_gpu_emulation(
            const int ncol, const int nlay, const int ngpt,
            const int ngas, const int nflav, const int ntemp, const int neta,
            const int nminor,
            const int nminork,
            const int idx_h2o, const int idx_tropo,
            const int *gpoint_flavor,
            const Float *kminor,
            const int *minor_limits_gpt,
            const Bool *minor_scales_with_density,
            const Bool *scale_by_complement,
            const int *idx_minor,
            const int *idx_minor_scaling,
            const int *kminor_start,
            const Float *play,
            const Float *tlay,
            const Float *col_gas,
            const Float *fminor,
            const int *jeta,
            const int *jtemp,
            const Bool *tropo,
            Float *tau,
            Float *tau_minor) {

        for (int bly = 0; bly < n_block_y; ++bly) {
            for (int blz = 0; blz < n_block_z; ++blz) {
                for (int thy = 0; thy < block_size_y; ++thy) {
                    for (int thz = 0; thz < block_size_z; ++thz) {

                        const int ilay = bly * block_size_y + thy;
                        const int icol = blz * block_size_z + thz;

                        if ((icol < ncol) && (ilay < nlay)) {
                            const int idx_collay = icol + ilay * ncol;
                            if (tropo[idx_collay] == idx_tropo) {

                                for (int imnr = 0; imnr < nminor; ++imnr) {
                                    Float scaling = Float(0.);

                                    const int ncl = ncol * nlay;
                                    scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                                    if (minor_scales_with_density[imnr]) {
                                        const Float PaTohPa = 0.01;
                                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];

                                        if (idx_minor_scaling[imnr] > 0) {
                                            const int idx_collaywv = icol + ilay * ncol + idx_h2o * ncl;
                                            Float vmr_fact = Float(1.) / col_gas[idx_collay];
                                            Float dry_fact =
                                                    Float(1.) / (Float(1.) + col_gas[idx_collaywv] * vmr_fact);

                                            if (scale_by_complement[imnr])
                                                scaling *= (Float(1.) -
                                                            col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] *
                                                            vmr_fact *
                                                            dry_fact);
                                            else
                                                scaling *=
                                                        col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] *
                                                        vmr_fact *
                                                        dry_fact;
                                        }
                                    }

                                    const int gpt_start = minor_limits_gpt[2 * imnr] - 1;
                                    const int gpt_end = minor_limits_gpt[2 * imnr + 1];
                                    const int gpt_offs = 1 - idx_tropo;
                                    const int iflav = gpoint_flavor[2 * gpt_start + gpt_offs] - 1;

                                    const int idx_fcl2 = 2 * 2 * (icol + ilay * ncol + iflav * ncol * nlay);
                                    const int idx_fcl1 = 2 * (icol + ilay * ncol + iflav * ncol * nlay);

                                    const Float *kfminor = &fminor[idx_fcl2];
                                    const Float *kin = &kminor[0];

                                    const int j0 = jeta[idx_fcl1];
                                    const int j1 = jeta[idx_fcl1 + 1];
                                    const int kjtemp = jtemp[idx_collay];
                                    const int band_gpt = gpt_end - gpt_start;
                                    const int gpt_offset = kminor_start[imnr] - 1;

                                    for (int igpt = 0; igpt < band_gpt; ++igpt) {
                                        Float ltau_minor = kfminor[0] * kin[(kjtemp - 1) + (j0 - 1) * ntemp +
                                                                            (igpt + gpt_offset) * ntemp * neta] +
                                                           kfminor[1] *
                                                           kin[(kjtemp - 1) + j0 * ntemp +
                                                               (igpt + gpt_offset) * ntemp * neta] +
                                                           kfminor[2] *
                                                           kin[kjtemp + (j1 - 1) * ntemp +
                                                               (igpt + gpt_offset) * ntemp * neta] +
                                                           kfminor[3] *
                                                           kin[kjtemp + j1 * ntemp +
                                                               (igpt + gpt_offset) * ntemp * neta];

                                        const int idx_out = icol + ilay * ncol + (igpt + gpt_start) * ncol * nlay;
                                        tau[idx_out] += ltau_minor * scaling;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template<int block_size_y, int block_size_z, int n_block_y, int n_block_z>
    void serial_kernel(
            const int ncol, const int nlay, const int ngpt,
            const int ngas, const int nflav, const int ntemp, const int neta,
            const int nminor,
            const int nminork,
            const int idx_h2o, const int idx_tropo,
            const int *gpoint_flavor, // 2 * ngpt
            const Float *kminor,
            const int *minor_limits_gpt,
            const Bool *minor_scales_with_density,
            const Bool *scale_by_complement,
            const int *idx_minor,
            const int *idx_minor_scaling,
            const int *kminor_start,
            const Float *play,
            const Float *tlay,
            const Float *col_gas,
            const Float *fminor,
            const int *jeta,
            const int *jtemp,
            const Bool *tropo,
            Float *tau,
            Float *tau_minor) {

        const int gpt_offs = 1 - idx_tropo;
        const Float *kin = &kminor[0];
        const Float PaTohPa = 0.01;
        const int ncl = ncol * nlay;


        int maxicol = n_block_z * (block_size_z);
        int maxilay = n_block_y * (block_size_y);
        if (maxilay > nlay) maxilay = nlay;
        if (maxicol > ncol) maxicol = ncol;

        const int limit = maxilay * (maxicol);

        for (int idx_collay = 0; idx_collay < limit; ++idx_collay) {
            {
                const int idx_collaywv = idx_collay + idx_h2o * ncl;

                Float vmr_fact = Float(1.) / col_gas[idx_collay];
                Float dry_fact =
                        Float(1.) / (Float(1.) + col_gas[idx_collaywv] * vmr_fact);

                if (tropo[idx_collay] == idx_tropo) {

                    for (int imnr = 0; imnr < nminor; ++imnr) {

                        Float scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                        if (minor_scales_with_density[imnr]) {
                            scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];

                            if (idx_minor_scaling[imnr] > 0) {


                                if (scale_by_complement[imnr])
                                    scaling *= (Float(1.) -
                                                col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] *
                                                vmr_fact *
                                                dry_fact);
                                else
                                    scaling *=
                                            col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] *
                                            vmr_fact *
                                            dry_fact;
                            }
                        }

                        const int gpt_start = minor_limits_gpt[2 * imnr] - 1;
                        const int gpt_end = minor_limits_gpt[2 * imnr + 1];

                        const int iflav = gpoint_flavor[2 * gpt_start + gpt_offs] - 1;

                        const int idx_fcl2 = 2 * 2 * (idx_collay + iflav * ncl);
                        const int idx_fcl1 = 2 * (idx_collay + iflav * ncl);

                        const Float *kfminor = &fminor[idx_fcl2];

                        const int j0 = jeta[idx_fcl1];
                        const int j1 = jeta[idx_fcl1 + 1];
                        const int kjtemp = jtemp[idx_collay];
                        const int band_gpt = gpt_end - gpt_start;
                        const int gpt_offset = kminor_start[imnr] - 1;

                        for (int igpt = 0; igpt < band_gpt; ++igpt) {
                            Float ltau_minor = kfminor[0] * kin[(kjtemp - 1) + (j0 - 1) * ntemp +
                                                                (igpt + gpt_offset) * ntemp * neta] +
                                               kfminor[1] *
                                               kin[(kjtemp - 1) + j0 * ntemp +
                                                   (igpt + gpt_offset) * ntemp * neta] +
                                               kfminor[2] *
                                               kin[kjtemp + (j1 - 1) * ntemp +
                                                   (igpt + gpt_offset) * ntemp * neta] +
                                               kfminor[3] *
                                               kin[kjtemp + j1 * ntemp +
                                                   (igpt + gpt_offset) * ntemp * neta];

                            const int idx_out = idx_collay + (igpt + gpt_start) * ncl;
                            tau[idx_out] += ltau_minor * scaling;
                        }
                    }
                }
            }

        }
    }

    template<int block_size_y, int block_size_z, int n_block_y, int n_block_z>
    void serial_kernel_mp(
            const int ncol, const int nlay, const int ngpt,
            const int ngas, const int nflav, const int ntemp, const int neta,
            const int nminor,
            const int nminork,
            const int idx_h2o, const int idx_tropo,
            const int *gpoint_flavor, // 2 * ngpt
            const Float *kminor,
            const int *minor_limits_gpt,
            const Bool *minor_scales_with_density,
            const Bool *scale_by_complement,
            const int *idx_minor,
            const int *idx_minor_scaling,
            const int *kminor_start,
            const Float *play,
            const Float *tlay,
            const Float *col_gas,
            const Float *fminor,
            const int *jeta,
            const int *jtemp,
            const Bool *tropo,
            Float *tau,
            Float *tau_minor) {

        const int gpt_offs = 1 - idx_tropo;
        const Float *kin = &kminor[0];
        const Float PaTohPa = 0.01;
        const int ncl = ncol * nlay;


        int maxicol = n_block_z * (block_size_z);
        int maxilay = n_block_y * (block_size_y);
        if (maxilay > nlay) maxilay = nlay;
        if (maxicol > ncol) maxicol = ncol;

        const int limit = maxilay * (maxicol);

#pragma omp parallel for num_threads(8)
        for (int idx_collay = 0; idx_collay < limit; ++idx_collay) {
            const int idx_collaywv = idx_collay + idx_h2o * ncl;

            Float vmr_fact = Float(1.) / col_gas[idx_collay];
            Float dry_fact =
                    Float(1.) / (Float(1.) + col_gas[idx_collaywv] * vmr_fact);

            if (tropo[idx_collay] == idx_tropo) {

                for (int imnr = 0; imnr < nminor; ++imnr) {

                    Float scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                    if (minor_scales_with_density[imnr]) {
                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];

                        if (idx_minor_scaling[imnr] > 0) {


                            if (scale_by_complement[imnr])
                                scaling *= (Float(1.) -
                                            col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] *
                                            vmr_fact *
                                            dry_fact);
                            else
                                scaling *=
                                        col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] *
                                        vmr_fact *
                                        dry_fact;
                        }
                    }

                    const int gpt_start = minor_limits_gpt[2 * imnr] - 1;
                    const int gpt_end = minor_limits_gpt[2 * imnr + 1];

                    const int iflav = gpoint_flavor[2 * gpt_start + gpt_offs] - 1;

                    const int idx_fcl2 = 2 * 2 * (idx_collay + iflav * ncl);
                    const int idx_fcl1 = 2 * (idx_collay + iflav * ncl);

                    const Float *kfminor = &fminor[idx_fcl2];

                    const int j0 = jeta[idx_fcl1];
                    const int j1 = jeta[idx_fcl1 + 1];
                    const int kjtemp = jtemp[idx_collay];
                    const int band_gpt = gpt_end - gpt_start;
                    const int gpt_offset = kminor_start[imnr] - 1;

                    for (int igpt = 0; igpt < band_gpt; ++igpt) {
                        Float ltau_minor = kfminor[0] * kin[(kjtemp - 1) + (j0 - 1) * ntemp +
                                                            (igpt + gpt_offset) * ntemp * neta] +
                                           kfminor[1] *
                                           kin[(kjtemp - 1) + j0 * ntemp +
                                               (igpt + gpt_offset) * ntemp * neta] +
                                           kfminor[2] *
                                           kin[kjtemp + (j1 - 1) * ntemp +
                                               (igpt + gpt_offset) * ntemp * neta] +
                                           kfminor[3] *
                                           kin[kjtemp + j1 * ntemp +
                                               (igpt + gpt_offset) * ntemp * neta];

                        const int idx_out = idx_collay + (igpt + gpt_start) * ncl;

                        tau[idx_out] += ltau_minor * scaling;
                    }
                }
            }
        }

    }

    template<typename T>
    T *alloc_and_copy(const T *device_src, size_t size) {
        T *dst = static_cast<T *>(malloc(sizeof(T) * size));
        cudaMemcpy(dst, device_src, sizeof(T) * size, cudaMemcpyDeviceToHost);
        return dst;
    }


    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const int *gpoint_flavor,
            const int *band_lims_gpt,
            const Float *kmajor,
            const Float *kminor_lower,
            const Float *kminor_upper,
            const int *minor_limits_gpt_lower,
            const int *minor_limits_gpt_upper,
            const Bool *minor_scales_with_density_lower,
            const Bool *minor_scales_with_density_upper,
            const Bool *scale_by_complement_lower,
            const Bool *scale_by_complement_upper,
            const int *idx_minor_lower,
            const int *idx_minor_upper,
            const int *idx_minor_scaling_lower,
            const int *idx_minor_scaling_upper,
            const int *kminor_start_lower,
            const int *kminor_start_upper,
            const Bool *tropo,
            const Float *col_mix, const Float *fmajor,
            const Float *fminor, const Float *play,
            const Float *tlay, const Float *col_gas,
            const int *jeta, const int *jtemp,
            const int *jpress,
            Float *tau) {
        Tuner_map &tunings = Tuner::get_map();

        dim3 grid_gpu_maj(ngpt, nlay, ncol);
        dim3 block_gpu_maj;

        if (tunings.count("gas_optical_depths_major_kernel") == 0) {
            Float *tau_tmp = Tools_gpu::allocate_gpu<Float>(ngpt * nlay * ncol);
            std::tie(grid_gpu_maj, block_gpu_maj) =
                    tune_kernel_compile_time<Gas_optical_depths_major_kernel>(
                            "gas_optical_depths_major_kernel",
                            dim3(ngpt, nlay, ncol),
                            std::integer_sequence < unsigned int, 1, 2, 4, 8, 16, 24, 32, 48, 64 > {},
                            std::integer_sequence < unsigned int, 1, 2, 4 > {},
                            std::integer_sequence < unsigned int, 8, 16, 24, 32, 48, 64, 96, 128, 256 > {},
                            ncol, nlay, nband, ngpt,
                            nflav, neta, npres, ntemp,
                            gpoint_flavor, band_lims_gpt,
                            kmajor, col_mix, fmajor, jeta,
                            tropo, jtemp, jpress,
                            tau_tmp);

            Tools_gpu::free_gpu<Float>(tau_tmp);

            tunings["gas_optical_depths_major_kernel"].first = grid_gpu_maj;
            tunings["gas_optical_depths_major_kernel"].second = block_gpu_maj;
        } else {
            grid_gpu_maj = tunings["gas_optical_depths_major_kernel"].first;
            block_gpu_maj = tunings["gas_optical_depths_major_kernel"].second;
        }

        run_kernel_compile_time<Gas_optical_depths_major_kernel>(
                std::integer_sequence < unsigned int, 1, 2, 4, 8, 16, 24, 32, 48, 64 > {},
                std::integer_sequence < unsigned int, 1, 2, 4 > {},
                std::integer_sequence < unsigned int, 8, 16, 24, 32, 48, 64, 96, 128, 256 > {},
                grid_gpu_maj, block_gpu_maj,
                ncol, nlay, nband, ngpt,
                nflav, neta, npres, ntemp,
                gpoint_flavor, band_lims_gpt,
                kmajor, col_mix, fmajor, jeta,
                tropo, jtemp, jpress,
                tau);

#define SERIAL_KERNEL true

#ifndef SERIAL_KERNEL
        printf("parallel kernel\n");
        // Lower
        int idx_tropo = 1;

        dim3 grid_gpu_min_1(1, 42, 8);
        dim3 block_gpu_min_1(8, 1, 16);

        gas_optical_depths_minor_kernel<8, 1, 16><<<grid_gpu_min_1, block_gpu_min_1>>>(
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nminorlower,
                nminorklower,
                idx_h2o, idx_tropo,
                gpoint_flavor,
                kminor_lower,
                minor_limits_gpt_lower,
                minor_scales_with_density_lower,
                scale_by_complement_lower,
                idx_minor_lower,
                idx_minor_scaling_lower,
                kminor_start_lower,
                play, tlay, col_gas,
                fminor, jeta, jtemp,
                tropo, tau, nullptr);

        // Upper
        idx_tropo = 0;

        dim3 grid_gpu_min_2(1, 42, 4);
        dim3 block_gpu_min_2(8, 1, 32);

        gas_optical_depths_minor_kernel<8, 1, 32><<<grid_gpu_min_2, block_gpu_min_2>>>(
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nminorupper,
                nminorkupper,
                idx_h2o, idx_tropo,
                gpoint_flavor,
                kminor_upper,
                minor_limits_gpt_upper,
                minor_scales_with_density_upper,
                scale_by_complement_upper,
                idx_minor_upper,
                idx_minor_scaling_upper,
                kminor_start_upper,
                play, tlay, col_gas,
                fminor, jeta, jtemp,
                tropo, tau, nullptr);
#else
        /// copy from device memory ////////////////////
        int *h_gpoint_flavor = alloc_and_copy<int>(gpoint_flavor, 2 * ngpt);
        Float *h_kminor_lower = alloc_and_copy<Float>(kminor_lower, ntemp * neta * nminorklower);
        int *h_minor_limits_gpt_lower = alloc_and_copy<int>(minor_limits_gpt_lower, 2 * nminorlower);
        Bool *h_minor_scales_with_density_lower = alloc_and_copy<Bool>(minor_scales_with_density_lower, nminorlower);
        Bool *h_scale_by_complement_lower = alloc_and_copy<Bool>(scale_by_complement_lower, nminorlower);
        int *h_idx_minor_lower = alloc_and_copy<int>(idx_minor_lower, nminorlower);
        int *h_idx_minor_scaling_lower = alloc_and_copy<int>(idx_minor_scaling_lower, nminorlower);
        int *h_kminor_start_lower = alloc_and_copy<int>(kminor_start_lower, nminorlower);
        Float *h_play = alloc_and_copy<Float>(play, ncol * nlay);
        Float *h_tlay = alloc_and_copy<Float>(tlay, ncol * nlay);
        Float *h_col_gas = alloc_and_copy<Float>(col_gas, ncol * nlay * (ngas + 1));
        Float *h_fminor = alloc_and_copy<Float>(fminor, 2 * 2 * ncol * nlay * nflav);
        int *h_jeta = alloc_and_copy<int>(jeta, 2 * ncol * nlay * nflav);
        int *h_jtemp = alloc_and_copy<int>(jtemp, ncol * nlay);
        Bool *h_tropo = alloc_and_copy<Bool>(tropo, ncol * nlay);
        Float *h_tau = alloc_and_copy<Float>(tau, ncol * nlay * ngpt);
        Float *h_kminor_upper = alloc_and_copy<Float>(kminor_upper, ntemp * neta * nminorkupper);
        int *h_minor_limits_gpt_upper = alloc_and_copy<int>(minor_limits_gpt_upper, 2 * nminorupper);
        Bool *h_minor_scales_with_density_upper = alloc_and_copy<Bool>(minor_scales_with_density_upper, nminorupper);
        Bool *h_scale_by_complement_upper = alloc_and_copy<Bool>(scale_by_complement_upper, nminorupper);
        int *h_idx_minor_upper = alloc_and_copy<int>(idx_minor_upper, nminorupper);
        int *h_idx_minor_scaling_upper = alloc_and_copy<int>(idx_minor_scaling_upper, nminorupper);
        int *h_kminor_start_upper = alloc_and_copy<int>(kminor_start_upper, nminorupper);
        ///////////////////

#if SERIAL_KERNEL == true

        printf("serial mp kernel\n");

        serial_kernel_mp<1, 16, 42, 8>(
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nminorlower,
                nminorklower,
                idx_h2o, 1,
                h_gpoint_flavor,
                h_kminor_lower,
                h_minor_limits_gpt_lower,
                h_minor_scales_with_density_lower,
                h_scale_by_complement_lower,
                h_idx_minor_lower,
                h_idx_minor_scaling_lower,
                h_kminor_start_lower,
                h_play, h_tlay, h_col_gas,
                h_fminor, h_jeta, h_jtemp,
                h_tropo, h_tau, nullptr);

        serial_kernel_mp<1, 32, 42, 4>(ncol, nlay, ngpt,
                                       ngas, nflav, ntemp, neta,
                                       nminorupper,
                                       nminorkupper,
                                       idx_h2o, 0,
                                       h_gpoint_flavor,
                                       h_kminor_upper,
                                       h_minor_limits_gpt_upper,
                                       h_minor_scales_with_density_upper,
                                       h_scale_by_complement_upper,
                                       h_idx_minor_upper,
                                       h_idx_minor_scaling_upper,
                                       h_kminor_start_upper,
                                       h_play, h_tlay, h_col_gas,
                                       h_fminor, h_jeta, h_jtemp,
                                       h_tropo, h_tau, nullptr);
#else
        printf("serial kernel\n");

        serial_kernel<1, 16, 42, 8>(
                ncol, nlay, ngpt,
                ngas, nflav, ntemp, neta,
                nminorlower,
                nminorklower,
                idx_h2o, 1,
                h_gpoint_flavor,
                h_kminor_lower,
                h_minor_limits_gpt_lower,
                h_minor_scales_with_density_lower,
                h_scale_by_complement_lower,
                h_idx_minor_lower,
                h_idx_minor_scaling_lower,
                h_kminor_start_lower,
                h_play, h_tlay, h_col_gas,
                h_fminor, h_jeta, h_jtemp,
                h_tropo, h_tau, nullptr);

        serial_kernel<1, 32, 42, 4>(ncol, nlay, ngpt,
                                          ngas, nflav, ntemp, neta,
                                          nminorupper,
                                          nminorkupper,
                                          idx_h2o, 0,
                                          h_gpoint_flavor,
                                          h_kminor_upper,
                                          h_minor_limits_gpt_upper,
                                          h_minor_scales_with_density_upper,
                                          h_scale_by_complement_upper,
                                          h_idx_minor_upper,
                                          h_idx_minor_scaling_upper,
                                          h_kminor_start_upper,
                                          h_play, h_tlay, h_col_gas,
                                          h_fminor, h_jeta, h_jtemp,
                                          h_tropo, h_tau, nullptr);
#endif
        cudaMemcpy(tau, h_tau, sizeof(Float) * ncol * nlay * ngpt, cudaMemcpyHostToDevice);
#endif
    }


    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Float *tlay,
            const Float *tlev,
            const Float *tsfc,
            const int sfc_lay,
            const Float *fmajor,
            const int *jeta,
            const Bool *tropo,
            const int *jtemp,
            const int *jpress,
            const int *gpoint_bands,
            const int *band_lims_gpt,
            const Float *pfracin,
            const Float temp_ref_min, const Float totplnk_delta,
            const Float *totplnk,
            const int *gpoint_flavor,
            Float *sfc_src,
            Float *lay_src,
            Float *lev_src_inc,
            Float *lev_src_dec,
            Float *sfc_src_jac) {
        Tuner_map &tunings = Tuner::get_map();

        const Float delta_Tsurf = Float(1.);

        const int block_gpt = 16;
        const int block_lay = 4;
        const int block_col = 2;

        const int grid_gpt = ngpt / block_gpt + (ngpt % block_gpt > 0);
        const int grid_lay = nlay / block_lay + (nlay % block_lay > 0);
        const int grid_col = ncol / block_col + (ncol % block_col > 0);

        dim3 grid_gpu(grid_gpt, grid_lay, grid_col);
        dim3 block_gpu(block_gpt, block_lay, block_col);

        if (tunings.count("Planck_source_kernel") == 0) {
            std::tie(grid_gpu, block_gpu) = tune_kernel(
                    "Planck_source_kernel",
                    dim3(ngpt, nlay, ncol),
                    {1, 2, 4},
                    {1, 2},
                    {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256},
                    Planck_source_kernel,
                    ncol, nlay, nbnd, ngpt,
                    nflav, neta, npres, ntemp, nPlanckTemp,
                    tlay, tlev, tsfc, sfc_lay,
                    fmajor, jeta, tropo, jtemp,
                    jpress, gpoint_bands, band_lims_gpt,
                    pfracin, temp_ref_min, totplnk_delta,
                    totplnk, gpoint_flavor,
                    delta_Tsurf, sfc_src, lay_src,
                    lev_src_inc, lev_src_dec,
                    sfc_src_jac);

            tunings["Planck_source_kernel"].first = grid_gpu;
            tunings["Planck_source_kernel"].second = block_gpu;
        } else {
            grid_gpu = tunings["Planck_source_kernel"].first;
            block_gpu = tunings["Planck_source_kernel"].second;
        }

        Planck_source_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                nflav, neta, npres, ntemp, nPlanckTemp,
                tlay, tlev, tsfc, sfc_lay,
                fmajor, jeta, tropo, jtemp,
                jpress, gpoint_bands, band_lims_gpt,
                pfracin, temp_ref_min, totplnk_delta,
                totplnk, gpoint_flavor,
                delta_Tsurf,
                sfc_src, lay_src,
                lev_src_inc, lev_src_dec,
                sfc_src_jac);
    }

}
