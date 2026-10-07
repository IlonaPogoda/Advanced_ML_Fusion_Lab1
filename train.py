from app.services import (
    TrainingService,
)


def main() -> None:

    service = (
        TrainingService()
    )

    result = (
        service.train_all()
    )

    print(
        "\nTraining finished.\n"
    )

    for (
        model_name,
        metrics
    ) in (
        result[
            "metrics"
        ].items()
    ):

        print(
            f"{model_name:>6}: "
            f"accuracy="
            f"{metrics['accuracy']:.4f}, "
            f"macro_f1="
            f"{metrics['macro_f1']:.4f}, "
            f"val_loss="
            f"{metrics['validation_loss']:.4f}"
        )


if __name__ == "__main__":
    main()