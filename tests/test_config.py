from datetime import date

from cordon import config


def test_week_zero_starts_on_the_policy_date_and_it_is_a_sunday():
    assert config.POLICY_START.weekday() == 6
    assert config.week_index(date(2025, 1, 5)) == 0
    assert config.week_index(date(2025, 1, 4)) == -1
    assert config.week_start(0) == date(2025, 1, 5)


def test_panel_windows_are_whole_years_of_weeks():
    assert config.FIRST_WEEK == -157
    assert config.PRIMARY_PRE_WEEK == -104


def test_last_week_is_complete_and_the_next_is_not():
    assert config.week_start(config.LAST_WEEK) == date(2026, 9, 20)
    assert config.week_start(config.LAST_WEEK + 1).month == 9
    assert config.week_start(config.LAST_WEEK + 1).day == 27


def test_christmas_stays_in_one_season_bin():
    bins = {
        config.week_index(date(y, 12, 25)) % config.SEASON_LENGTH for y in (2022, 2023, 2024, 2025)
    }
    assert len(bins) == 1


def test_months_covers_inclusive_range_across_years():
    produced = list(config.months((2024, 11), (2025, 2)))
    assert produced == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
