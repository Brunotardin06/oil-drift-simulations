from compare_spills_gif import generate_comparison_gif


class OutputService:
    """Generate output artifacts (images, gifs, summaries)."""

    def generate_comparison_gif(self, **kwargs):
        return generate_comparison_gif(**kwargs)

