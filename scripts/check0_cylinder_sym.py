import numpy as np

from check_cylinders_common import render_check


def main() -> None:
    render_check("check0_sym", (0.0, np.deg2rad(180.0), 0.0), 1, False)


if __name__ == "__main__":
    main()
