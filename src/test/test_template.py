from unittest import TestCase
from utils.template import parse_template


class Test(TestCase):
    def test_parse_template(self):
        self.assertEqual('111111test11111111 - 111111test11111111', parse_template('{{test}} - {{test}}', {'test': '111111test11111111'}))

