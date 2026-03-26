from np_shift import build_default_experiment, run


def main() -> None:
    config = build_default_experiment()
    print(run(config))


if __name__ == "__main__":
    main()
