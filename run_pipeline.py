from movie_pipeline import movie_review_pipeline

if __name__ == "__main__":
    movie_review_pipeline(
                    learning_rate=2e-05,
                    batch_size=16,
                    epochs=3,
                    dropout=0.3,
                    optimizer_name="Adam"
                )
