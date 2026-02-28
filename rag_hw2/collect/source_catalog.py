from __future__ import annotations

from pathlib import Path
import re

from rag_hw2.collect.crawler import CrawlConfig


COMMON_DENY_SUBSTRINGS = [
    "mailto:",
    "javascript:",
    "/wp-json/",
    "/feed",
    "/rss",
    "/tag/",
    "/author/",
    "/privacy",
    "/terms",
    "/login",
    "/signin",
    "/account",
    "/cart",
    "/checkout",
    "facebook.com",
    "instagram.com",
    "x.com/",
    "twitter.com",
    "youtube.com",
    "linkedin.com",
]

COMMON_DENY_REGEXES = [
    r"\.(?:jpg|jpeg|png|gif|webp|svg|ico)(?:\?.*)?$",
    r"\.(?:css|js|xml)(?:\?.*)?$",
    r"/amp(?:/|$)",
]


class SourceJob:
    def __init__(
        self,
        name,
        group,
        seed_urls,
        allowed_domains,
        max_pages= 120,
        max_depth= 5,
        allow_query= False,
        download_pdfs= True,
        sleep_sec= 0.6,
        url_allow_prefixes= None,
        url_allow_substrings= None,
        url_deny_substrings= None,
        url_deny_regexes= None,
        notes= None,
    ) :
        self.name = name
        self.group = group
        self.seed_urls = list(seed_urls)
        self.allowed_domains = list(allowed_domains)
        self.max_pages = max_pages
        depth = int(max_depth)
        if depth < 4:
            depth = 4
        if depth > 5:
            depth = 5
        self.max_depth = depth
        self.allow_query = allow_query
        self.download_pdfs = download_pdfs
        self.sleep_sec = sleep_sec
        self.url_allow_prefixes = list(url_allow_prefixes) if url_allow_prefixes else None
        self.url_allow_substrings = list(url_allow_substrings) if url_allow_substrings else None
        self.url_deny_substrings = list(url_deny_substrings) if url_deny_substrings else None
        self.url_deny_regexes = list(url_deny_regexes) if url_deny_regexes else None
        self.notes = notes

    def out_dir(self, out_root) :
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", self.name.strip()).strip("_").lower()
        return str(Path(out_root) / self.group / safe)

    def to_crawl_config(self, out_root, sleep_override= None) :
        deny_substrings = list(COMMON_DENY_SUBSTRINGS)
        if self.url_deny_substrings:
            deny_substrings.extend(self.url_deny_substrings)
        deny_regexes = list(COMMON_DENY_REGEXES)
        if self.url_deny_regexes:
            deny_regexes.extend(self.url_deny_regexes)
        return CrawlConfig(
            seed_urls=self.seed_urls,
            out_dir=self.out_dir(out_root),
            allowed_domains=self.allowed_domains,
            max_pages=self.max_pages,
            max_depth=self.max_depth,
            allow_query=self.allow_query,
            download_pdfs=self.download_pdfs,
            sleep_sec=self.sleep_sec if sleep_override is None else sleep_override,
            url_allow_prefixes=self.url_allow_prefixes,
            url_allow_substrings=self.url_allow_substrings,
            url_deny_substrings=deny_substrings,
            url_deny_regexes=deny_regexes,
        )


def _sports_deny() :
    return [
        "/news",
        "/video",
        "/videos",
        "/watch",
        "/scores",
        "/stats",
        "/gameday",
        "/podcast",
        "/tickets",
        "/shop",
        "/community",
        "/fantasy",
        "/betting",
    ]


def recommended_source_jobs() :
    jobs = []

    # General / history
    jobs.extend(
        [
            SourceJob(
                name="wikipedia_pittsburgh_history",
                group="general",
                seed_urls=[
                    "https://en.wikipedia.org/wiki/Pittsburgh",
                    "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
                ],
                allowed_domains=["wikipedia.org"],
                max_pages=80,
                max_depth=1,
                url_allow_prefixes=["https://en.wikipedia.org/wiki/"],
                url_deny_substrings=["/wiki/Special:", "/wiki/Help:", "/wiki/Talk:"],
            ),
            SourceJob(
                name="city_of_pittsburgh_home",
                group="general",
                seed_urls=["https://www.pittsburghpa.gov/Home"],
                allowed_domains=["pittsburghpa.gov"],
                max_pages=80,
                max_depth=2,
            ),
            SourceJob(
                name="city_tax_regulations",
                group="general",
                seed_urls=["https://pittsburghpa.gov/finance/tax-forms"],
                allowed_domains=["pittsburghpa.gov"],
                max_pages=120,
                max_depth=2,
                download_pdfs=True,
                url_allow_substrings=["pittsburghpa.gov/finance", "/tax", "/regulation", ".pdf"],
            ),
            SourceJob(
                name="city_budget_2025_pdf",
                group="general",
                seed_urls=[
                    "https://www.pittsburghpa.gov/files/assets/city/v/4/omb/documents/operating-budgets/2025-operating-budget.pdf"
                ],
                allowed_domains=["pittsburghpa.gov"],
                max_pages=5,
                max_depth=0,
                download_pdfs=True,
            ),
            SourceJob(
                name="britannica_pittsburgh",
                group="general",
                seed_urls=["https://www.britannica.com/place/Pittsburgh"],
                allowed_domains=["britannica.com"],
                max_pages=30,
                max_depth=1,
                url_allow_substrings=["britannica.com/place/Pittsburgh"],
            ),
            SourceJob(
                name="visit_pittsburgh_general",
                group="general",
                seed_urls=[
                    "https://www.visitpittsburgh.com/",
                    "https://www.visitpittsburgh.com/things-to-do/",
                ],
                allowed_domains=["visitpittsburgh.com"],
                max_pages=160,
                max_depth=2,
            ),
            SourceJob(
                name="cmu_about_history",
                group="general",
                seed_urls=[
                    "https://www.cmu.edu/about/",
                    "https://www.cmu.edu/about/history.html",
                ],
                allowed_domains=["cmu.edu"],
                max_pages=80,
                max_depth=2,
                url_allow_substrings=["cmu.edu/about"],
            ),
        ]
    )

    # Events
    jobs.extend(
        [
            SourceJob(
                name="pittsburgh_events_calendar",
                group="events",
                seed_urls=[
                    "https://pittsburgh.events",
                    "https://pittsburgh.events/calendar",
                ],
                allowed_domains=["pittsburgh.events"],
                max_pages=200,
                max_depth=3,
                allow_query=True,
                url_deny_substrings=["/news", "/blog", "/contact", "/advertise"],
            ),
            SourceJob(
                name="downtown_pittsburgh_events",
                group="events",
                seed_urls=["https://downtownpittsburgh.com/events/"],
                allowed_domains=["downtownpittsburgh.com"],
                max_pages=150,
                max_depth=3,
            ),
            SourceJob(
                name="pgh_citypaper_events",
                group="events",
                seed_urls=["https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d"],
                allowed_domains=["pghcitypaper.com"],
                max_pages=150,
                max_depth=2,
                allow_query=True,
                url_allow_substrings=["/pittsburgh/EventSearch", "/event", "/EventSearch", "pghcitypaper.com"],
            ),
            SourceJob(
                name="cmu_events_calendar",
                group="events",
                seed_urls=["https://events.cmu.edu"],
                allowed_domains=["events.cmu.edu"],
                max_pages=180,
                max_depth=3,
                allow_query=True,
            ),
            SourceJob(
                name="cmu_campus_events_page",
                group="events",
                seed_urls=["https://www.cmu.edu/engage/alumni/events/campus/index.html"],
                allowed_domains=["cmu.edu"],
                max_pages=80,
                max_depth=2,
                url_allow_substrings=["cmu.edu/engage/alumni/events/campus"],
            ),
        ]
    )

    # Music / culture
    jobs.extend(
        [
            SourceJob(
                name="pittsburgh_symphony",
                group="culture",
                seed_urls=["https://www.pittsburghsymphony.org/"],
                allowed_domains=["pittsburghsymphony.org"],
                max_pages=140,
                max_depth=2,
            ),
            SourceJob(
                name="pittsburgh_opera",
                group="culture",
                seed_urls=["https://pittsburghopera.org/"],
                allowed_domains=["pittsburghopera.org"],
                max_pages=140,
                max_depth=2,
            ),
            SourceJob(
                name="trustarts",
                group="culture",
                seed_urls=["https://trustarts.org/"],
                allowed_domains=["trustarts.org"],
                max_pages=180,
                max_depth=3,
            ),
            SourceJob(
                name="carnegie_museums",
                group="culture",
                seed_urls=["https://carnegiemuseums.org/"],
                allowed_domains=["carnegiemuseums.org"],
                max_pages=160,
                max_depth=3,
            ),
            SourceJob(
                name="heinz_history_center",
                group="culture",
                seed_urls=["https://www.heinzhistorycenter.org/"],
                allowed_domains=["heinzhistorycenter.org"],
                max_pages=160,
                max_depth=3,
            ),
            SourceJob(
                name="the_frick_pittsburgh",
                group="culture",
                seed_urls=["https://www.thefrickpittsburgh.org/"],
                allowed_domains=["thefrickpittsburgh.org"],
                max_pages=160,
                max_depth=3,
            ),
            SourceJob(
                name="visit_pittsburgh_arts_culture",
                group="culture",
                seed_urls=[
                    "https://www.visitpittsburgh.com/things-to-do/arts-culture/",
                    "https://www.visitpittsburgh.com/things-to-do/arts-culture/museums/",
                ],
                allowed_domains=["visitpittsburgh.com"],
                max_pages=120,
                max_depth=2,
            ),
        ]
    )

    # Food
    jobs.extend(
        [
            SourceJob(
                name="visit_pittsburgh_food_festivals",
                group="food",
                seed_urls=["https://www.visitpittsburgh.com/events-festivals/food-festivals/"],
                allowed_domains=["visitpittsburgh.com"],
                max_pages=100,
                max_depth=2,
            ),
            SourceJob(
                name="picklesburgh",
                group="food",
                seed_urls=["https://www.picklesburgh.com/"],
                allowed_domains=["picklesburgh.com"],
                max_pages=100,
                max_depth=2,
            ),
            SourceJob(
                name="pittsburgh_taco_fest",
                group="food",
                seed_urls=["https://www.pghtacofest.com/"],
                allowed_domains=["pghtacofest.com"],
                max_pages=100,
                max_depth=2,
            ),
            SourceJob(
                name="pittsburgh_restaurant_week",
                group="food",
                seed_urls=["https://pittsburghrestaurantweek.com/"],
                allowed_domains=["pittsburghrestaurantweek.com"],
                max_pages=120,
                max_depth=2,
            ),
            SourceJob(
                name="little_italy_days",
                group="food",
                seed_urls=["https://littleitalydays.com/"],
                allowed_domains=["littleitalydays.com"],
                max_pages=100,
                max_depth=2,
            ),
            SourceJob(
                name="banana_split_fest",
                group="food",
                seed_urls=["https://bananasplitfest.com/"],
                allowed_domains=["bananasplitfest.com"],
                max_pages=100,
                max_depth=2,
            ),
        ]
    )

    # Sports (focused, avoid news/scores)
    jobs.extend(
        [
            SourceJob(
                name="visit_pittsburgh_sports_teams",
                group="sports",
                seed_urls=["https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/"],
                allowed_domains=["visitpittsburgh.com"],
                max_pages=80,
                max_depth=2,
            ),
            SourceJob(
                name="pirates_site",
                group="sports",
                seed_urls=["https://www.mlb.com/pirates"],
                allowed_domains=["mlb.com"],
                max_pages=120,
                max_depth=2,
                url_allow_substrings=["mlb.com/pirates"],
                url_deny_substrings=_sports_deny(),
            ),
            SourceJob(
                name="steelers_site",
                group="sports",
                seed_urls=["https://www.steelers.com/"],
                allowed_domains=["steelers.com"],
                max_pages=120,
                max_depth=2,
                url_deny_substrings=_sports_deny(),
            ),
            SourceJob(
                name="penguins_site",
                group="sports",
                seed_urls=["https://www.nhl.com/penguins/"],
                allowed_domains=["nhl.com"],
                max_pages=120,
                max_depth=2,
                url_allow_substrings=["nhl.com/penguins"],
                url_deny_substrings=_sports_deny(),
            ),
        ]
    )

    return jobs


def filter_jobs(
    jobs,
    groups= None,
    include_names= None,
    exclude_names= None,
) :
    out = jobs
    if groups:
        group_set = {g.lower() for g in groups}
        out = [j for j in out if j.group.lower() in group_set]
    if include_names:
        inc = {x.lower() for x in include_names}
        out = [j for j in out if j.name.lower() in inc]
    if exclude_names:
        exc = {x.lower() for x in exclude_names}
        out = [j for j in out if j.name.lower() not in exc]
    return out
