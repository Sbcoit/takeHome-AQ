"""Tests for generation components."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.generation.topics import TopicSampler, TopicContext
from src.models.schemas import PhysicsTopic


class TestTopicSampler:
    """Tests for TopicSampler."""

    @pytest.fixture
    def sampler(self):
        """Fixture for topic sampler."""
        return TopicSampler()

    def test_sample_returns_topic_context(self, sampler):
        """Test that sample returns a TopicContext."""
        context = sampler.sample()
        assert isinstance(context, TopicContext)
        assert isinstance(context.topic, PhysicsTopic)
        assert context.subtopic
        assert context.difficulty_modifier

    def test_sample_with_diversity(self, sampler):
        """Test that diversity tracking works."""
        contexts = [sampler.sample(prefer_diverse=True) for _ in range(10)]

        # Should have sampled from multiple topics
        topics = set(c.topic for c in contexts)
        assert len(topics) > 1

    def test_sample_batch(self, sampler):
        """Test batch sampling."""
        contexts = sampler.sample_batch(5)
        assert len(contexts) == 5
        for context in contexts:
            assert isinstance(context, TopicContext)

    def test_sample_batch_with_coverage(self, sampler):
        """Test batch sampling with topic coverage."""
        # Request enough samples to cover all topics
        n_topics = len(PhysicsTopic)
        contexts = sampler.sample_batch(n_topics + 5, ensure_topic_coverage=True)

        assert len(contexts) == n_topics + 5

        # All topics should be represented
        topics = set(c.topic for c in contexts)
        assert len(topics) == n_topics

    def test_coverage_stats(self, sampler):
        """Test coverage statistics."""
        for _ in range(10):
            sampler.sample()

        stats = sampler.get_coverage_stats()
        assert "topics_used" in stats
        assert "total_generated" in stats
        assert stats["total_generated"] == 10

    def test_reset_diversity(self, sampler):
        """Test resetting diversity tracking."""
        for _ in range(5):
            sampler.sample()

        assert sampler.get_coverage_stats()["total_generated"] == 5

        sampler.reset_diversity_tracking()

        assert sampler.get_coverage_stats()["total_generated"] == 0

    def test_topic_context_to_prompt_string(self):
        """Test TopicContext prompt string generation."""
        context = TopicContext(
            topic=PhysicsTopic.QUANTUM_MECHANICS,
            subtopic="Perturbation theory",
            difficulty_modifier="with degenerate states",
            additional_context="Consider first-order corrections.",
        )

        prompt = context.to_prompt_string()

        assert "Perturbation theory" in prompt
        assert "Quantum Mechanics" in prompt
        assert "degenerate states" in prompt
        assert "first-order corrections" in prompt

    def test_all_topics_have_subtopics(self, sampler):
        """Test that all physics topics have subtopics defined."""
        for topic in PhysicsTopic:
            subtopics = sampler.SUBTOPICS.get(topic, [])
            assert len(subtopics) > 0, f"Topic {topic} has no subtopics"

    def test_custom_topics_list(self):
        """Test sampler with custom topics list."""
        custom_topics = [PhysicsTopic.QUANTUM_MECHANICS, PhysicsTopic.ELECTROMAGNETISM]
        sampler = TopicSampler(topics=custom_topics)

        for _ in range(20):
            context = sampler.sample()
            assert context.topic in custom_topics
