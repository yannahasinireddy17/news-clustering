import os

from news_pipeline import train_ag_news_pipeline


if __name__ == "__main__":
	base_dir = os.path.dirname(os.path.abspath(__file__))
	result = train_ag_news_pipeline(
		os.path.join(base_dir, "datasets", "train.csv"),
		os.path.join(base_dir, "datasets", "test.csv"),
		model_dir=os.path.join(base_dir, "models"),
	)
	print("AG News K-Means training complete")
	print(result)