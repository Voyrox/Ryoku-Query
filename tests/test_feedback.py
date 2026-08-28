import sqlite3
import tempfile
import unittest
from pathlib import Path

from feedback import FeedbackStore, FeedbackView


class FeedbackStoreTests(unittest.TestCase):
    def test_records_answer_feedback_and_returns_it_for_later_export(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "nero-feedback.sqlite3")
            store.record_answer(
                answer_message_id=101,
                request_message_id=100,
                requester_id=7,
            )

            outcome = store.record_feedback(
                answer_message_id=101,
                actor_id=7,
                verdict="correct",
            )

            self.assertEqual(outcome, "recorded")
            self.assertEqual(
                store.recent_feedback(),
                [
                    {
                        "answer_message_id": 101,
                        "request_message_id": 100,
                        "requester_id": 7,
                        "actor_id": 7,
                        "verdict": "correct",
                    }
                ],
            )

    def test_replaces_one_requesters_verdict_without_creating_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "nero-feedback.sqlite3")
            store.record_answer(
                answer_message_id=101,
                request_message_id=100,
                requester_id=7,
            )
            store.record_feedback(101, 7, "incorrect")

            outcome = store.record_feedback(101, 7, "correct")

            self.assertEqual(outcome, "updated")
            self.assertEqual(len(store.recent_feedback()), 1)
            self.assertEqual(store.recent_feedback()[0]["verdict"], "correct")

    def test_rejects_feedback_from_someone_other_than_the_requester(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "nero-feedback.sqlite3")
            store.record_answer(
                answer_message_id=101,
                request_message_id=100,
                requester_id=7,
            )

            for verdict in ("correct", "partially_correct", "incorrect"):
                with self.subTest(verdict=verdict):
                    self.assertEqual(store.record_feedback(101, 8, verdict), "not_requester")
            self.assertEqual(store.recent_feedback(), [])

    def test_records_partially_correct_feedback_for_the_requester(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "nero-feedback.sqlite3")
            store.record_answer(
                answer_message_id=101,
                request_message_id=100,
                requester_id=7,
            )

            outcome = store.record_feedback(101, 7, "partially_correct")

            self.assertEqual(outcome, "recorded")
            self.assertEqual(store.recent_feedback()[0]["verdict"], "partially_correct")
            self.assertEqual(store.record_feedback(101, 8, "partially_correct"), "not_requester")

    def test_migrates_existing_feedback_database_for_partially_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nero-feedback.sqlite3"
            connection = sqlite3.connect(path)
            try:
                with connection:
                    connection.executescript(
                        """
                        CREATE TABLE answers (
                            answer_message_id INTEGER PRIMARY KEY,
                            request_message_id INTEGER NOT NULL,
                            requester_id INTEGER NOT NULL
                        );
                        CREATE TABLE feedback (
                            answer_message_id INTEGER NOT NULL,
                            actor_id INTEGER NOT NULL,
                            verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect')),
                            PRIMARY KEY (answer_message_id, actor_id),
                            FOREIGN KEY (answer_message_id) REFERENCES answers(answer_message_id)
                        );
                        """
                    )
                    connection.execute("INSERT INTO answers VALUES (101, 100, 7)")
                    connection.execute("INSERT INTO feedback VALUES (101, 7, 'correct')")
            finally:
                connection.close()

            store = FeedbackStore(path)

            self.assertEqual(store.record_feedback(101, 7, "partially_correct"), "updated")
            self.assertEqual(store.recent_feedback()[0]["verdict"], "partially_correct")

    def test_view_offers_correct_partial_and_incorrect_buttons(self):
        with tempfile.TemporaryDirectory() as directory:
            view = FeedbackView(FeedbackStore(Path(directory) / "nero-feedback.sqlite3"))

        self.assertEqual(
            [item.label for item in view.children],
            ["Correct", "Partially correct", "Incorrect"],
        )
        self.assertEqual(
            [item.custom_id for item in view.children],
            [
                "nero_feedback:correct",
                "nero_feedback:partially_correct",
                "nero_feedback:incorrect",
            ],
        )


if __name__ == "__main__":
    unittest.main()
