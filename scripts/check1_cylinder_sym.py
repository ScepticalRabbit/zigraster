import numpy as np

from check_cylinders_common import render_check


def main() -> None:
    render_check("check1_sym", (0.0, np.deg2rad(-25.0), 0.0), 4, True)


if __name__ == "__main__":
    main()
