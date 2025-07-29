from scope.model import ScOPE


model = ScOPE(
    compressor_name="bz2",
    compression_metric="ncd",
    compression_level=6,
    use_best_sigma=True,
    model_type='ot',
    matching_method_name='jaccard'
)


sample_uk = ["The movie was awsome"]

kw_samples = {
    'positive': [
        "This movie was absolutely fantastic! I loved every minute of it and would definitely watch it again.",
        "Amazing cinematography and brilliant acting. One of the best films I've seen this year.",
        "Incredible story with outstanding performances. Highly recommend to everyone.",
        "Perfect blend of drama and comedy. The director did an exceptional job.",
        "Beautifully crafted film with memorable characters and excellent dialogue."
    ],
    'negative': [
        "This was one of the worst movies I've ever seen. Complete waste of time and money.",
        "Terrible acting, boring plot, and poor direction. I couldn't wait for it to end.",
        "Absolutely dreadful film with no redeeming qualities whatsoever.",
        "The worst cinematography and dialogue I've encountered. Avoid at all costs.",
        "Painfully boring and poorly executed. Two hours of my life I'll never get back."
    ],
}

result = model(sample_uk, kw_samples)

print(
    next(result)
)

print(model)