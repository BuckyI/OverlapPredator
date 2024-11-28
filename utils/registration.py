from typing import Optional

import numpy as np
import small_gicp


def GICP_registration(source: np.ndarray, target: np.ndarray, init_T: Optional[np.ndarray] = None):
    """
    inputs:
        source: np.ndarray (N, 3)
        target: np.ndarray (N, 3)
    return:
        transformation: np.ndarray (4, 4)
        registration_result: small_gicp.RegistrationResult
    """
    T = init_T or np.eye(4)
    for s in (0.25, 0.1, 0.02):
        result = small_gicp.align(
            target,
            source,
            init_T_target_source=T,
            registration_type="GICP",
            downsampling_resolution=s,
            max_correspondence_distance=2 * s,
        )
        T = result.T_target_source
    return result.T_target_source, result
