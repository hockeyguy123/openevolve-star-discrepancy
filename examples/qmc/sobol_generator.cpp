#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <numeric> // For std::accumulate if used, or bit_cast in C++20
#include <algorithm> // For std::min

// --- Constants ---
const int MAX_BITS = 30; // Matched to successful Python MAX_BITS_PY
// MAX_S_VAL_STORAGE might not be strictly needed if DimensionParameters.m_i is sized well.
// For Sobol, s is usually small enough that MAX_BITS is a safe upper bound for m_i array.
const int MAX_MI_ELEMENTS = MAX_BITS; // Max number of m_i values we might need for a dimension

struct DimensionParameters {
    int s_degree_from_file; // 's' from Joe & Kuo file
    uint32_t a_jk_from_file; // 'a' from Joe & Kuo file (coeffs a_1..a_{s-1})
    uint32_t m_i_initial_unscaled[MAX_MI_ELEMENTS]; // Initial UN SCALED m_i values
};

// --- Helper Functions ---

// Calculates bit_length similar to Python's int.bit_length()
inline int get_bit_length(uint32_t n) {
    if (n == 0) return 0;
    int count = 0;
    // For uint32_t, max 32 iterations
    for (int i = 31; i >= 0; --i) {
        if ((n >> i) & 1U) {
            count = i + 1;
            break;
        }
    }
    return count;
}

// Converts Joe & Kuo (s, a) to SciPy's polynomial integer P(x) = (1 A_1 ... A_{s-1} 1)_2
// P(x) = x^s + A_1 x^{s-1} + ... + A_{s-1} x + 1
inline uint32_t get_scipy_poly_from_jk(int s_degree_param, uint32_t a_jk_param) {
    if (s_degree_param == 0) return 1U; // poly x^0 = 1 (SciPy uses P(x)=1 for s=0, effectively degree 0)
                                     // Though often P(x)=x+1 is used for s=0 in some contexts, SciPy poly is just 1.
                                     // The python code returns (1<<1)|1 for s=1, a=0. Let's match python.
                                     // Python code: if s_degree == 0: return DTYPE_UINT(1) # poly x^0+1 = 1 -- this is actually P(x)=1
                                     // if s_degree == 1: return DTYPE_UINT( (1 << 1) | 1 ) # 0b11 --> P(x)=x+1
    if (s_degree_param == 1) return (1U << 1) | 1U; // Matches python for s=1 (poly x+1)

    // For s_degree_param >= 2:
    // p_val has bit s_degree_param set (for x^s_degree_param)
    // and bit 0 set (for constant 1)
    // a_jk_param (coeffs A_1 to A_{s_degree_param-1}) are shifted to fit between bit s_degree_param-1 and bit 1.
    uint32_t p_val = (1U << s_degree_param);      // x^s_degree_param term
    p_val |= (a_jk_param << 1);                  // A_1..A_{s_degree_param-1} shifted to positions s_degree_param-1..1
    p_val |= 1U;                               // Constant 1 term (bit 0)
    return p_val;
}

// Helper to get the k-th bit of an integer (0-indexed k, LSB is bit 0)
inline uint32_t get_kth_bit_from_int(uint32_t n, int k) {
    return (n >> k) & 1U;
}


extern "C" {
    void generate_sobol_points(
        int n_points,
        int n_dimensions,
        const DimensionParameters* input_sobol_params_for_dims_2_onwards, // Array of size n_dimensions-1
        double* output_points, // Flat array of size n_points * n_dimensions
        const uint32_t* final_ltm_elements_flat, // Flat D*B*B tensor, already processed (tril, diag=1)
        const uint32_t* final_digital_shifts     // Flat D array of final shift integers
    ) {
        if (n_points == 0 || n_dimensions == 0) {
            return;
        }

        std::vector<std::vector<uint32_t>> V_direction_numbers(n_dimensions, std::vector<uint32_t>(MAX_BITS));
        std::vector<std::vector<uint32_t>> m_unscaled_matrix(n_dimensions, std::vector<uint32_t>(MAX_BITS));

        // --- Step 1: Generate Unscaled m_k and then V_k (direction numbers) ---
        for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
            if (dim_idx == 0) { // First Sobol dimension
                for (int k_idx = 0; k_idx < MAX_BITS; ++k_idx) {
                    m_unscaled_matrix[dim_idx][k_idx] = 1U;
                }
            } else { // Subsequent Sobol dimensions
                if (input_sobol_params_for_dims_2_onwards == nullptr && n_dimensions > 1) {
                     // Handle error: missing parameters
                    for(int i=0; i < n_points*n_dimensions; ++i) output_points[i] = -999.0; 
                    return;
                }
                const DimensionParameters& current_params_struct = input_sobol_params_for_dims_2_onwards[dim_idx - 1];
                int s_from_file = current_params_struct.s_degree_from_file;
                uint32_t a_from_file = current_params_struct.a_jk_from_file;

                uint32_t poly_p_int = get_scipy_poly_from_jk(s_from_file, a_from_file);
                
                int s_poly_deg = 0; // True degree of poly_p_int
                if (poly_p_int > 0) {
                    s_poly_deg = get_bit_length(poly_p_int) - 1;
                }
                // Python code had a warning here if s_poly_deg != s_from_file, C++ can too if needed.

                // Initialize first s_poly_deg unscaled m_k values from input m_i list
                int num_initial_m_to_use = std::min({s_poly_deg, MAX_MI_ELEMENTS, MAX_BITS});
                for (int k_init_m = 0; k_init_m < num_initial_m_to_use; ++k_init_m) {
                    m_unscaled_matrix[dim_idx][k_init_m] = current_params_struct.m_i_initial_unscaled[k_init_m];
                }
                 // Fill remaining with 0 up to s_poly_deg if m_i list was shorter
                for (int k_init_m = num_initial_m_to_use; k_init_m < s_poly_deg && k_init_m < MAX_BITS; ++k_init_m) {
                    m_unscaled_matrix[dim_idx][k_init_m] = 0U; // As per Python's np.zeros initialization
                }


                // Bratley & Fox style recurrence for m_j (unscaled)
                for (int j_unscaled_m_idx = s_poly_deg; j_unscaled_m_idx < MAX_BITS; ++j_unscaled_m_idx) {
                    uint32_t new_m_val = m_unscaled_matrix[dim_idx][j_unscaled_m_idx - s_poly_deg];
                    uint32_t current_power_of_2_factor = 1U; 
                    
                    for (int k_poly_loop = 0; k_poly_loop < s_poly_deg; ++k_poly_loop) {
                        current_power_of_2_factor <<= 1; // Becomes 2^1, 2^2, ..., 2^s_poly_deg
                        
                        // Coeff for x^{s_poly_deg - (k_poly_loop+1)} term in poly.
                        // (poly_p_int >> (s_poly_deg - 1 - k_poly_loop)) & 1U
                        uint32_t coeff_is_1 = (poly_p_int >> (s_poly_deg - 1 - k_poly_loop)) & 1U;
                        
                        if (coeff_is_1) {
                            int idx_m_jm_k = j_unscaled_m_idx - k_poly_loop - 1;
                            if (idx_m_jm_k >=0) { // Ensure index is valid
                               new_m_val ^= (current_power_of_2_factor * m_unscaled_matrix[dim_idx][idx_m_jm_k]);
                            }
                        }
                    }
                    m_unscaled_matrix[dim_idx][j_unscaled_m_idx] = new_m_val;
                }
            }

            // Scale all unscaled m_k to get the final V_k matrix for the current dimension
            for (int k_scale_idx = 0; k_scale_idx < MAX_BITS; ++k_scale_idx) {
                V_direction_numbers[dim_idx][k_scale_idx] = 
                    m_unscaled_matrix[dim_idx][k_scale_idx] << (MAX_BITS - (k_scale_idx + 1));
            }
        }
        
        // --- Step 2: Apply Linear Matrix Scramble (LMS) if final_ltm_elements_flat is provided ---
        // Assumes final_ltm_elements_flat is D x B x B, already processed (tril, diag=1) by Python
        if (final_ltm_elements_flat != nullptr) { 
            for (int d_lms = 0; d_lms < n_dimensions; ++d_lms) {
                const uint32_t* ltm_for_current_dim_start = final_ltm_elements_flat + (d_lms * MAX_BITS * MAX_BITS);
                
                for (int j_v_orig_idx = 0; j_v_orig_idx < MAX_BITS; ++j_v_orig_idx) { 
                    uint32_t V_orig_val = V_direction_numbers[d_lms][j_v_orig_idx];
                    uint32_t transformed_Vj_acc = 0U;
                    
                    // bit_of_new_V_LSB is the LSB index of the output V's bit
                    for (int bit_of_new_V_LSB = 0; bit_of_new_V_LSB < MAX_BITS; ++bit_of_new_V_LSB) {
                        // LTM row index is MSB-indexed from the LSB output bit index
                        int ltm_row_MSB_idx = MAX_BITS - 1 - bit_of_new_V_LSB;
                        
                        uint32_t dot_product_mod_2 = 0;
                        // k_input_bit_LSB is the LSB index of the input V_orig_val's bit
                        for (int k_input_bit_LSB = 0; k_input_bit_LSB < MAX_BITS; ++k_input_bit_LSB) {
                            // LTM column index is MSB-indexed from the LSB input bit index
                            int ltm_col_MSB_idx = MAX_BITS - 1 - k_input_bit_LSB;
                            
                            uint32_t ltm_element = ltm_for_current_dim_start[ltm_row_MSB_idx * MAX_BITS + ltm_col_MSB_idx];
                            uint32_t input_bit = get_kth_bit_from_int(V_orig_val, k_input_bit_LSB);
                            dot_product_mod_2 += ltm_element * input_bit;
                        }
                        dot_product_mod_2 %= 2; 
                        
                        if (dot_product_mod_2 == 1) {
                            transformed_Vj_acc |= (1U << bit_of_new_V_LSB);
                        }
                    }
                    V_direction_numbers[d_lms][j_v_orig_idx] = transformed_Vj_acc;
                }
            }
        }

        // --- Step 3: Point generation part with digital shift ---
        // SciPy's random_base2(m) normalizes by 2**self.bits, not 2**m.
        // self.bits is MAX_BITS here.
        const double denominator = static_cast<double>(1ULL << MAX_BITS); 

        for (int pt_idx = 0; pt_idx < n_points; ++pt_idx) {
            uint32_t gray_code_pt = static_cast<uint32_t>(pt_idx); // For clarity
            uint32_t gray_code = gray_code_pt ^ (gray_code_pt >> 1);

            for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
                uint32_t sobol_int_val_raw = 0U;
                for (int k_bit = 0; k_bit < MAX_BITS; ++k_bit) {
                    if ((gray_code >> k_bit) & 1U) {
                        sobol_int_val_raw ^= V_direction_numbers[dim_idx][k_bit];
                    }
                }
                
                uint32_t final_full_resolution_sobol_int = sobol_int_val_raw;
                if (final_digital_shifts != nullptr) { 
                    final_full_resolution_sobol_int ^= final_digital_shifts[dim_idx];
                }
                
                // The m parameter (from n_points = 2^m) does not affect the precision of the output points'
                // individual coordinate values, only the number of points and thus the Gray codes used.
                // Normalization is always by 2^MAX_BITS.
                output_points[pt_idx * n_dimensions + dim_idx] = 
                    static_cast<double>(final_full_resolution_sobol_int) / denominator;
            }
        }
    }
} // extern "C"









// #include <vector>
// #include <cmath>   // For std::pow (though not directly used in the V generation logic)
// #include <cstdint> // For uint32_t
// #include <stdexcept> // For std::runtime_error (optional, for safety)
// #include <numeric> // For std::iota (potentially, or manual loop)

// // Define the maximum number of bits for Sobol sequence generation
// const int MAX_BITS = 32;
// // Define the maximum 's' value (degree of polynomial) that m_i can hold
// const int MAX_S_VAL_STORAGE = 32; // Max size of the m_i array in the struct

// struct DimensionParameters {
//     int s;          // Degree of the primitive polynomial
//     uint32_t a;     // Integer representation of coefficients (a_1, ..., a_{s-1})
//     uint32_t m_i[MAX_S_VAL_STORAGE]; // Initial UN SCALED direction numerators (m_1, ..., m_s)
// };

// extern "C" {
//     // Keeping the function name as per your last snippet
//     void generate_sobol_points( 
//         int n_points,
//         int n_dimensions, // Total number of dimensions to generate
//         const DimensionParameters* input_sobol_params, // Array of (n_dimensions - 1) params for dims 2 onwards
//         double* output_points,
//         const uint32_t* scramble_masks
//     ) {
//         if (n_points == 0) { 
//             return;
//         }
//         if (n_dimensions == 0) {
//             return;
//         }

//         std::vector<std::vector<uint32_t>> V_final_scaled(n_dimensions, std::vector<uint32_t>(MAX_BITS));

//         for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
//             if (dim_idx == 0) {
//                 // ----- Hardcode the first Sobol dimension (dim_idx = 0) -----
//                 // V_final_scaled[0][k] = m_{k+1} * 2^(MAX_BITS - (k+1)), where m_{k+1} = 1
//                 for (int k_v_idx = 0; k_v_idx < MAX_BITS; ++k_v_idx) {
//                     V_final_scaled[dim_idx][k_v_idx] = 1U << (MAX_BITS - (k_v_idx + 1));
//                 }
//             } else {
//                 // ----- Compute subsequent Sobol dimensions (dim_idx > 0) to match SciPy's _initialize_v -----
//                 // Parameters for Sobol dimension (dim_idx+1) are taken from input_sobol_params[dim_idx - 1].
//                 if (input_sobol_params == nullptr && n_dimensions > 1) {
//                      // Simplified error handling for C context; consider returning an error code.
//                     for(int i=0; i < n_points*n_dimensions; ++i) output_points[i] = -1.0; // Indicate error
//                     return;
//                 }

//                 const DimensionParameters& current_params = input_sobol_params[dim_idx - 1];
//                 int s_degree = current_params.s;      // Degree of polynomial for this dimension
//                 uint32_t a_from_file = current_params.a; // (a_1 a_2 ... a_{s-1})_2

//                 // 1. Construct SciPy-like polynomial integer `p_val = (1 a_1 a_2 ... a_{s-1} 1)_2`
//                 //    Degree `s_degree` means polynomial is x^s_degree + a_1 x^{s_degree-1} + ... + a_{s_degree-1} x + 1
//                 //    (where a_{s_degree} is 1).
//                 //    `a_from_file` contains (s_degree-1) coefficients a_1 to a_{s_degree-1}.
//                 uint32_t p_val = (1U << s_degree); // x^s_degree term (bit at position s_degree)
//                 p_val |= (a_from_file << 1);    // a_1 to a_{s_degree-1} terms (bits s_degree-1 down to 1)
//                 p_val |= 1U;                    // Constant term '1' (bit at position 0)
//                                                 // Note: SciPy's poly[d] is this p_val.
//                                                 // bit_length(p_val) - 1 will give s_degree.

//                 // Temporary vector to store unscaled direction numerators (m_i values for this dimension)
//                 std::vector<uint32_t> m_numerators_unscaled(MAX_BITS);

//                 // 2. Initialize first s_degree unscaled numerators from input current_params.m_i
//                 for (int k = 0; k < s_degree; ++k) {
//                     if (k >= MAX_BITS) break; // Should not happen if s_degree <= MAX_S_VAL_STORAGE <= MAX_BITS
//                     m_numerators_unscaled[k] = current_params.m_i[k];
//                 }

//                 // 3. Generate remaining unscaled numerators using SciPy's recurrence
//                 //    j is 0-indexed for m_numerators_unscaled, from s_degree to MAX_BITS-1
//                 //    This calculates m_numerators_unscaled[s_degree] through m_numerators_unscaled[MAX_BITS-1]
//                 for (int j_idx = s_degree; j_idx < MAX_BITS; ++j_idx) {
//                     uint32_t new_m_val = m_numerators_unscaled[j_idx - s_degree]; // Term m_{j-s}
//                     uint32_t pow2_term = 1U; 
//                     // k_poly_coeff_loop iterates from 0 to s_degree-1, representing coeffs c_1 to c_s
//                     // (where c_s is from p_val's bit 0, c_1 from p_val's bit s-1)
//                     for (int k_poly_coeff_loop = 0; k_poly_coeff_loop < s_degree; ++k_poly_coeff_loop) {
//                         pow2_term <<= 1; // Becomes 2, 4, ..., 2^s_degree

//                         // Extract coefficient c_{k_poly_coeff_loop+1} from p_val.
//                         // Coeff for 2^1 * m_{j-1} is c_1 (from p_val bit s_degree-1)
//                         // Coeff for 2^s * m_{j-s} is c_s (from p_val bit 0)
//                         // SciPy: (p >> (m - 1 - k)) & 1. Here m=s_degree. k=k_poly_coeff_loop.
//                         if ((p_val >> (s_degree - 1 - k_poly_coeff_loop)) & 1U) {
//                             // Check index bounds for m_numerators_unscaled:
//                             // j_idx - k_poly_coeff_loop - 1 must be >= 0
//                             // Smallest j_idx is s_degree. Largest k_poly_coeff_loop is s_degree-1.
//                             // s_degree - (s_degree-1) - 1 = 0. So index is valid.
//                             if ((j_idx - k_poly_coeff_loop - 1) < MAX_BITS) { // Redundant if loop is correct
//                                 new_m_val ^= (pow2_term * m_numerators_unscaled[j_idx - k_poly_coeff_loop - 1]);
//                             }
//                         }
//                     }
//                     m_numerators_unscaled[j_idx] = new_m_val;
//                 }

//                 // 4. Populate final V_final_scaled matrix by scaling all m_numerators, matching SciPy's observed scaling
//                 // V_final_scaled[dim_idx][k_0idx] = m_numerators_unscaled[k_0idx] * (2^(MAX_BITS - 1 - k_0idx))
//                 uint32_t pow2_final_scale = 1U;
//                 for (int k_col_loop = 0; k_col_loop < MAX_BITS; ++k_col_loop) { // k_col_loop from 0 to MAX_BITS-1
//                     // Column index to scale: (MAX_BITS - 1 - k_col_loop), goes from MAX_BITS-1 down to 0
//                     int current_col_idx = MAX_BITS - 1 - k_col_loop;
//                     if (current_col_idx < MAX_BITS) { // Redundant if loop is correct
//                          V_final_scaled[dim_idx][current_col_idx] = m_numerators_unscaled[current_col_idx] * pow2_final_scale;
//                     }
//                     if (k_col_loop < MAX_BITS -1) { // Avoid overflow on last shift if pow2_final_scale is uint32_t
//                         pow2_final_scale <<= 1;
//                     }
//                 }
//             }
//         } // End of V matrix calculation loop

//         // --- Point generation part (remains unchanged) ---
//         const double denominator = static_cast<double>(1ULL << MAX_BITS);

//         for (int pt_idx = 0; pt_idx < n_points; ++pt_idx) {
//             uint32_t gray_code = static_cast<uint32_t>(pt_idx) ^ (static_cast<uint32_t>(pt_idx) >> 1);
//             for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
//                 uint32_t sobol_int = 0;
//                 for (int k_bit = 0; k_bit < MAX_BITS; ++k_bit) {
//                     if ((gray_code >> k_bit) & 1) {
//                         sobol_int ^= V_final_scaled[dim_idx][k_bit];
//                     }
//                 }
//                 uint32_t scrambled_int = sobol_int ^ scramble_masks[dim_idx];
//                 output_points[pt_idx * n_dimensions + dim_idx] = static_cast<double>(scrambled_int) / denominator;
//             }
//         }
//     }
// } // extern "C"






// #include <vector>
// #include <cmath>   // For std::pow (though not directly used in the V generation logic)
// #include <cstdint> // For uint32_t
// #include <stdexcept> // For std::out_of_range (optional, for safety)

// // Define the maximum number of bits for Sobol sequence generation
// const int MAX_BITS = 32;
// // Define the maximum 's' value (degree of polynomial) that m_i can hold
// const int MAX_S_VAL_STORAGE = 32; // Max size of the m_i array in the struct

// struct DimensionParameters {
//     int s;
//     uint32_t a;
//     uint32_t m_i[MAX_S_VAL_STORAGE];
// };

// extern "C" {
//     // Function name reflects the new behavior regarding input_sobol_params
//     void generate_sobol_points( 
//         int n_points,
//         int n_dimensions, // Total number of dimensions to generate
//         const DimensionParameters* input_sobol_params, // Array of (n_dimensions - 1) params for dims 2 onwards
//         double* output_points,
//         const uint32_t* scramble_masks
//     ) {
//         if (n_points == 0) { 
//             return;
//         }
//         if (n_dimensions == 0) {
//             return;
//         }

//         std::vector<std::vector<uint32_t>> V(n_dimensions, std::vector<uint32_t>(MAX_BITS));

//         for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
//             if (dim_idx == 0) {
//                 // ----- Hardcode the first Sobol dimension (dim_idx = 0) -----
//                 // V_i = m_i * 2^(MAX_BITS - i), where m_i = 1 for all i for the first dimension.
//                 for (int k_v_idx = 0; k_v_idx < MAX_BITS; ++k_v_idx) {
//                     V[dim_idx][k_v_idx] = 1U << (MAX_BITS - (k_v_idx + 1));
//                 }
//             } else {
//                 // ----- Compute subsequent Sobol dimensions (dim_idx > 0) -----
//                 // Parameters for Sobol dimension (dim_idx+1) are taken from input_sobol_params[dim_idx - 1].
//                 // E.g., for Sobol dim 2 (dim_idx=1), use input_sobol_params[0].
//                 //       for Sobol dim 3 (dim_idx=2), use input_sobol_params[1].
                
//                 // Safety check: ensure input_sobol_params is not null if we need to access it.
//                 // This assumes the caller provides an array of (n_dimensions - 1) elements.
//                 // if (input_sobol_params == nullptr) {
//                 //     // Handle error: parameters needed but not provided.
//                 //     // This could be an assertion, an exception, or logging an error.
//                 //     // For now, let's throw an exception as an example.
//                 //     // In a C API, you might return an error code or have a pre-condition.
//                 //     throw std::runtime_error("input_sobol_params is null for subsequent dimensions.");
//                 // }

//                 const DimensionParameters& params = input_sobol_params[dim_idx - 1];
//                 int s = params.s;
//                 uint32_t a = params.a;

//                 // Initialize the first s direction numbers in V from m_i
//                 for (int k = 0; k < s; ++k) {
//                     if (k >= MAX_BITS) break; 
//                     V[dim_idx][k] = params.m_i[k] << (MAX_BITS - (k + 1));
//                 }

//                 // Generate remaining direction numbers in V using the recurrence
//                 for (int k = s; k < MAX_BITS; ++k) {
//                     V[dim_idx][k] = V[dim_idx][k - s] ^ (V[dim_idx][k - s] >> s);
//                     for (int l = 0; l < s - 1; ++l) {
//                         if ((a >> (s - 1 - (l + 1))) & 1) {
//                             V[dim_idx][k] ^= (V[dim_idx][k - (l + 1)] >> (l + 1));
//                         }
//                     }
//                 }
//             }
//         } // End of V matrix calculation loop

//         // --- Point generation part (remains unchanged) ---
//         const double denominator = static_cast<double>(1ULL << MAX_BITS);

//         for (int pt_idx = 0; pt_idx < n_points; ++pt_idx) {
//             uint32_t gray_code = static_cast<uint32_t>(pt_idx) ^ (static_cast<uint32_t>(pt_idx) >> 1);
//             for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
//                 uint32_t sobol_int = 0;
//                 for (int k_bit = 0; k_bit < MAX_BITS; ++k_bit) {
//                     if ((gray_code >> k_bit) & 1) {
//                         sobol_int ^= V[dim_idx][k_bit];
//                     }
//                 }
//                 uint32_t scrambled_int = sobol_int ^ scramble_masks[dim_idx];
//                 output_points[pt_idx * n_dimensions + dim_idx] = static_cast<double>(scrambled_int) / denominator;
//             }
//         }
//     }
// } // extern "C"




// #include <vector>
// #include <cmath>   // For std::pow
// #include <cstdint> // For uint32_t

// // Define the maximum number of bits for Sobol sequence generation
// const int MAX_BITS = 32;
// // Define the maximum 's' value (degree of polynomial) that m_i can hold
// const int MAX_S_VAL_STORAGE = 32; // Max size of the m_i array in the struct

// struct DimensionParameters {
//     int s;
//     uint32_t a;
//     uint32_t m_i[MAX_S_VAL_STORAGE];
// };

// extern "C" {
//     void generate_sobol_points( // Renamed to emphasize no checks
//         int n_points,
//         int n_dimensions,
//         const DimensionParameters* input_sobol_params,
//         double* output_points,
//         const uint32_t* scramble_masks
//     ) {
//         if (n_points == 0) { // Still useful to handle this trivial case
//             return;
//         }

//         std::vector<std::vector<uint32_t>> V(n_dimensions, std::vector<uint32_t>(MAX_BITS));

//         for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
//             const DimensionParameters& params = input_sobol_params[dim_idx];
//             int s = params.s;
//             uint32_t a = params.a;

//             for (int k = 0; k < s; ++k) {
//                 V[dim_idx][k] = params.m_i[k] << (MAX_BITS - (k + 1));
//             }

//             for (int k = s; k < MAX_BITS; ++k) {
//                 V[dim_idx][k] = V[dim_idx][k - s] ^ (V[dim_idx][k - s] >> s);
//                 for (int l = 0; l < s - 1; ++l) {
//                     if ((a >> (s - 1 - (l + 1))) & 1) {
//                         V[dim_idx][k] ^= (V[dim_idx][k - (l + 1)] >> (l + 1));
//                     }
//                 }
//             }
//         }

//         const double denominator = static_cast<double>(1ULL << MAX_BITS);

//         for (int pt_idx = 0; pt_idx < n_points; ++pt_idx) {
//             uint32_t gray_code = static_cast<uint32_t>(pt_idx) ^ (static_cast<uint32_t>(pt_idx) >> 1);
//             for (int dim_idx = 0; dim_idx < n_dimensions; ++dim_idx) {
//                 uint32_t sobol_int = 0;
//                 for (int k_bit = 0; k_bit < MAX_BITS; ++k_bit) {
//                     if ((gray_code >> k_bit) & 1) {
//                         sobol_int ^= V[dim_idx][k_bit];
//                     }
//                 }
//                 uint32_t scrambled_int = sobol_int ^ scramble_masks[dim_idx];
//                 output_points[pt_idx * n_dimensions + dim_idx] = static_cast<double>(scrambled_int) / denominator;
//             }
//         }
//     }
// } // extern "C"