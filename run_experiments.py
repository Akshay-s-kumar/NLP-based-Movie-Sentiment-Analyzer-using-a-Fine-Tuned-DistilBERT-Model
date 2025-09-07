from movie_pipeline import movie_review_pipeline

for lr in [2e-5, 3e-5]:
    for bs in [16, 32]:
        for dropout in [0.1, 0.3]:
            for optimizer in ["Adam", "AdamW"]:
                print(f"🚀 Running: lr={lr}, bs={bs}, dropout={dropout}, opt={optimizer}")
                movie_review_pipeline(
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=3,
                    dropout=dropout,
                    optimizer_name=optimizer
                )
