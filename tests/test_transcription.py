import unittest

from evaluate_transcription import fold_quotes, parse_transcription, summarize
from evaluate import compare


class TranscriptionTests(unittest.TestCase):
    def test_quote_fold_keeps_other_content(self):
        self.assertEqual(fold_quotes('\u201cIt\u2019s 2-3!\u201d'), '"It\'s 2-3!"')

    def sample(self):
        return '## page_001.txt\n\n```text\nA [UNCERTAIN: word]\nPage 1/2\n```\n## page_002.txt\n\n```text\nB\nPage 2/2\n```\n## Uncertain passages\n- page_001.txt - unclear word\n'

    def test_extract_and_keep_notes_out_of_reference(self):
        pages, flagged, inline = parse_transcription(self.sample(), 2)
        self.assertEqual(flagged, [1])
        self.assertEqual(inline, {1: 1})
        self.assertIn('[UNCERTAIN: word]', pages[1]['raw'])
        self.assertEqual(pages[1]['best_reading'], 'A word\nPage 1/2\n')
        self.assertNotIn('unclear', pages[1]['raw'])

    def test_reject_missing_duplicate_and_broken_fences(self):
        for bad in [self.sample().replace('page_002.txt', 'page_001.txt'), self.sample().replace('```text', '```', 1), self.sample().replace('Page 2/2', 'Page 3/2')]:
            with self.assertRaises(ValueError):
                parse_transcription(bad, 2)

    def test_reject_bad_uncertainty(self):
        for bad in [self.sample().replace('[UNCERTAIN: word]', '[UNCERTAIN: word'), self.sample() + '- page_003.txt - invalid\n']:
            with self.assertRaises(ValueError):
                parse_transcription(bad, 2)

    def test_pool_and_pair_same_pages(self):
        scores = {'a': [{'page': 1, 'score': compare('a b', 'a')}, {'page': 2, 'score': compare('c', 'c')}],
                  'b': [{'page': 1, 'score': compare('a b', 'a b')}, {'page': 2, 'score': compare('c', 'd')}]}
        report = summarize(scores, {1})
        self.assertEqual(report['engines']['a']['wer'], .5)
        self.assertEqual(report['engines']['b']['wer'], 0)
        self.assertEqual(report['paired_wer_differences'][0]['paired_95_interval_percentage_points'], [50, 50])


if __name__ == '__main__':
    unittest.main()
