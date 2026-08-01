import csv
import json
import logging

logger = logging.getLogger(__name__)

PAPER_FIELDS = ("paper_id", "senior_reviewers_needed", "reviewers_needed")
REVIEWER_FIELDS = (
    "reviewer_id",
    "max_load",
    "min_load",
    "seniority",
    "region",
    "authored_paper_ids",
    "coauthor_reviewer_ids",
)
SCORE_FIELDS = ("paper_id", "reviewer_id", "score")
CONSTRAINT_FIELDS = ("paper_id", "reviewer_id", "constraint")


def _csv_rows(path, expected_fields):
    """Yield validated CSV rows together with their one-based line number."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != expected_fields:
            raise ValueError(
                f"Invalid header in {path}: expected {','.join(expected_fields)}; "
                f"got {','.join(actual_fields) or '<missing>'}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Too many columns in {path} at line {line_number}")
            yield line_number, row


def _csv_values(path, expected_fields):
    """Yield validated positional CSV rows for high-volume pair files."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            actual_fields = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"Missing header in {path}") from exc
        if actual_fields != expected_fields:
            raise ValueError(
                f"Invalid header in {path}: expected {','.join(expected_fields)}; "
                f"got {','.join(actual_fields) or '<missing>'}"
            )
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(expected_fields):
                raise ValueError(
                    f"Invalid column count in {path} at line {line_number}: "
                    f"expected {len(expected_fields)}, got {len(values)}"
                )
            yield line_number, values


def _external_id(value, kind, path, line_number):
    external_id = value.strip()
    if not external_id:
        raise ValueError(f"Blank {kind} ID in {path} at line {line_number}")
    return external_id


def _integer(value, field, path, line_number):
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for {field} in {path} at line {line_number}: {value!r}"
        ) from exc


def _number(value, field, path, line_number):
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid number for {field} in {path} at line {line_number}: {value!r}"
        ) from exc


def _id_list(value, field, path, line_number):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON array for {field} in {path} at line {line_number}"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(
            f"{field} in {path} at line {line_number} must be a JSON array"
        )
    result = []
    for item in parsed:
        external_id = str(item).strip()
        if not external_id:
            raise ValueError(f"Blank ID in {field} in {path} at line {line_number}")
        result.append(external_id)
    return result


class InputInstance:
    """Load external dataset IDs and remap them to solver indices in memory."""

    def __init__(
        self,
        paper_info_file,
        reviewer_info_file,
        similarity_scores_file,
        bid_scores_file,
        constraints_file,
    ):
        self.bad_papers_list = []
        self.zero_capacity_reviewers = []
        self.reviewers_with_no_similarity_scores = []
        self.nonpositive_similarity_scores_skipped = 0

        self._load_papers(paper_info_file)
        self._load_reviewers(reviewer_info_file)
        self._validate_capacity()
        self._build_authorship_data()
        self._initialize_pairwise_data()
        self._load_similarity_scores(similarity_scores_file)
        self._finalize_bad_papers_and_reviewers()
        self._load_bid_scores(bid_scores_file)
        self._build_bidpaper_author_data()
        self._load_constraints(constraints_file)
        self._adjust_reviewer_loads_for_forced_assignments()

        logger.info(
            "Loaded %d papers, %d reviewers, and %d candidate pairs",
            self.np,
            self.nr,
            sum(len(reviewers) for reviewers in self.remained_r_for_p),
        )
        if self.nonpositive_similarity_scores_skipped:
            logger.info(
                "Skipped %d non-positive similarity-score rows",
                self.nonpositive_similarity_scores_skipped,
            )

    def _load_papers(self, paper_info_file):
        self.ellp = []
        self.ellp_sen = []
        self.paper_internal_to_external = []
        self.paper_external_to_internal = {}

        for line_number, row in _csv_rows(paper_info_file, PAPER_FIELDS):
            external_id = _external_id(
                row["paper_id"], "paper", paper_info_file, line_number
            )
            if external_id in self.paper_external_to_internal:
                raise ValueError(
                    f"Duplicate paper ID in {paper_info_file}: {external_id}"
                )

            senior_load = _integer(
                row["senior_reviewers_needed"],
                "senior_reviewers_needed",
                paper_info_file,
                line_number,
            )
            total_load = _integer(
                row["reviewers_needed"],
                "reviewers_needed",
                paper_info_file,
                line_number,
            )
            if senior_load < 0 or total_load < 0 or senior_load > total_load:
                raise ValueError(
                    f"Invalid reviewer requirements for paper {external_id}: "
                    f"senior={senior_load}, total={total_load}"
                )

            internal_id = len(self.paper_internal_to_external)
            self.paper_external_to_internal[external_id] = internal_id
            self.paper_internal_to_external.append(external_id)
            self.ellp_sen.append(senior_load)
            self.ellp.append(total_load)

        self.np = len(self.ellp)
        if self.np == 0:
            raise ValueError(f"No papers found in {paper_info_file}")

    def _load_reviewers(self, reviewer_info_file):
        reviewer_rows = list(_csv_rows(reviewer_info_file, REVIEWER_FIELDS))
        if not reviewer_rows:
            raise ValueError(f"No reviewers found in {reviewer_info_file}")

        self.reviewer_internal_to_external = []
        self.reviewer_external_to_internal = {}
        for line_number, row in reviewer_rows:
            external_id = _external_id(
                row["reviewer_id"], "reviewer", reviewer_info_file, line_number
            )
            if external_id in self.reviewer_external_to_internal:
                raise ValueError(
                    f"Duplicate reviewer ID in {reviewer_info_file}: {external_id}"
                )
            internal_id = len(self.reviewer_internal_to_external)
            self.reviewer_external_to_internal[external_id] = internal_id
            self.reviewer_internal_to_external.append(external_id)

        self.ellr = []
        self.min_ellr = []
        self.seniority = []
        self.region = []
        self.region_count = 0
        self.paperlist = []
        self.coauthorlist = []

        for reviewer_index, (line_number, row) in enumerate(reviewer_rows):
            max_load = _integer(
                row["max_load"], "max_load", reviewer_info_file, line_number
            )
            min_load = _integer(
                row["min_load"], "min_load", reviewer_info_file, line_number
            )
            seniority = _integer(
                row["seniority"], "seniority", reviewer_info_file, line_number
            )
            region = _integer(row["region"], "region", reviewer_info_file, line_number)
            if min_load < 0 or max_load < min_load:
                raise ValueError(
                    f"Invalid load bounds for reviewer {self.reviewer_id(reviewer_index)}: "
                    f"min={min_load}, max={max_load}"
                )
            if seniority < 0 or region < 0:
                raise ValueError(
                    f"Seniority and region must be non-negative for reviewer "
                    f"{self.reviewer_id(reviewer_index)}"
                )

            authored_papers = self._map_id_list(
                _id_list(
                    row["authored_paper_ids"],
                    "authored_paper_ids",
                    reviewer_info_file,
                    line_number,
                ),
                self.paper_external_to_internal,
                "paper",
                reviewer_info_file,
                line_number,
            )
            coauthors = self._map_id_list(
                _id_list(
                    row["coauthor_reviewer_ids"],
                    "coauthor_reviewer_ids",
                    reviewer_info_file,
                    line_number,
                ),
                self.reviewer_external_to_internal,
                "reviewer",
                reviewer_info_file,
                line_number,
            )

            self.ellr.append(max_load)
            self.min_ellr.append(min_load)
            self.seniority.append(seniority)
            self.region.append(region)
            self.region_count = max(self.region_count, region + 1)
            self.paperlist.append(authored_papers)
            self.coauthorlist.append(coauthors)
            if max_load == 0:
                self.zero_capacity_reviewers.append(reviewer_index)

        self.nr = len(self.ellr)

    @staticmethod
    def _map_id_list(external_ids, mapping, kind, path, line_number):
        internal_ids = []
        for external_id in external_ids:
            try:
                internal_ids.append(mapping[external_id])
            except KeyError as exc:
                raise ValueError(
                    f"Unknown {kind} ID {external_id!r} in {path} at line {line_number}"
                ) from exc
        return internal_ids

    def _pair_indices(self, paper_value, reviewer_value, path, line_number):
        paper_external_id = _external_id(paper_value, "paper", path, line_number)
        reviewer_external_id = _external_id(
            reviewer_value, "reviewer", path, line_number
        )
        try:
            paper = self.paper_external_to_internal[paper_external_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown paper ID {paper_external_id!r} in {path} at line {line_number}"
            ) from exc
        try:
            reviewer = self.reviewer_external_to_internal[reviewer_external_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown reviewer ID {reviewer_external_id!r} in {path} at line {line_number}"
            ) from exc
        return paper, reviewer

    def _validate_capacity(self):
        total_reviews_needed = sum(self.ellp)
        total_reviewer_capacity = sum(self.ellr)
        if total_reviews_needed > total_reviewer_capacity:
            raise ValueError(
                f"{total_reviews_needed} reviews needed, but only "
                f"{total_reviewer_capacity} reviewer load available"
            )

    def _build_authorship_data(self):
        self.authorship = [set() for _ in range(self.np)]
        self.authorlist = [[] for _ in range(self.np)]
        for reviewer, authored_papers in enumerate(self.paperlist):
            for paper in authored_papers:
                self.authorship[paper].add(reviewer)
                self.authorlist[paper].append(reviewer)

        self.coauthorship = [set(coauthors) for coauthors in self.coauthorlist]

    def _initialize_pairwise_data(self):
        self.s = [dict() for _ in range(self.np)]
        self.remained_r_for_p = [[] for _ in range(self.np)]
        self.remained_p_for_r = [[] for _ in range(self.nr)]
        self.constraint = [dict() for _ in range(self.np)]
        self.matched_this_stage = [set() for _ in range(self.np)]

    def _load_similarity_scores(self, similarity_scores_file):
        for line_number, (paper_value, reviewer_value, score_value) in _csv_values(
            similarity_scores_file, SCORE_FIELDS
        ):
            paper, reviewer = self._pair_indices(
                paper_value,
                reviewer_value,
                similarity_scores_file,
                line_number,
            )
            score = _number(
                score_value, "score", similarity_scores_file, line_number
            )
            if score <= 0:
                self.nonpositive_similarity_scores_skipped += 1
                continue
            if reviewer in self.s[paper]:
                raise ValueError(
                    f"Duplicate similarity pair in {similarity_scores_file} at line "
                    f"{line_number}: ({self.paper_id(paper)}, {self.reviewer_id(reviewer)})"
                )
            self.s[paper][reviewer] = score
            self.remained_r_for_p[paper].append(reviewer)
            self.remained_p_for_r[reviewer].append(paper)

    def _finalize_bad_papers_and_reviewers(self):
        for paper in range(self.np):
            if not self.s[paper]:
                self.bad_papers_list.append(paper)

        for reviewer in range(self.nr):
            if not self.remained_p_for_r[reviewer]:
                self.reviewers_with_no_similarity_scores.append(reviewer)

        self.zero_capacity_reviewer_mask = [False for _ in range(self.nr)]
        for reviewer in self.zero_capacity_reviewers:
            self.zero_capacity_reviewer_mask[reviewer] = True

    def _load_bid_scores(self, bid_scores_file):
        self.bid = [set() for _ in range(self.np)]
        self.bid_score = [dict() for _ in range(self.np)]
        self.biddedlist = [[] for _ in range(self.np)]
        self.bidlist = [[] for _ in range(self.nr)]

        for line_number, (paper_value, reviewer_value, score_value) in _csv_values(
            bid_scores_file, SCORE_FIELDS
        ):
            paper, reviewer = self._pair_indices(
                paper_value,
                reviewer_value,
                bid_scores_file,
                line_number,
            )
            if reviewer in self.bid_score[paper]:
                raise ValueError(
                    f"Duplicate bid pair in {bid_scores_file} at line {line_number}: "
                    f"({self.paper_id(paper)}, {self.reviewer_id(reviewer)})"
                )
            score = _number(score_value, "score", bid_scores_file, line_number)
            self.bid_score[paper][reviewer] = score
            if score > 0.0:
                self.bid[paper].add(reviewer)
                self.biddedlist[paper].append(reviewer)
                self.bidlist[reviewer].append(paper)

    def _build_bidpaper_author_data(self):
        self.bidpaper_author = [set() for _ in range(self.nr)]
        self.bidpaper_authorlist = [[] for _ in range(self.nr)]

        for reviewer in range(self.nr):
            for paper in self.bidlist[reviewer]:
                for author in self.authorlist[paper]:
                    if author not in self.bidpaper_author[reviewer]:
                        self.bidpaper_authorlist[reviewer].append(author)
                        self.bidpaper_author[reviewer].add(author)

    def _load_constraints(self, constraints_file):
        for line_number, (
            paper_value,
            reviewer_value,
            constraint_value,
        ) in _csv_values(constraints_file, CONSTRAINT_FIELDS):
            paper, reviewer = self._pair_indices(
                paper_value,
                reviewer_value,
                constraints_file,
                line_number,
            )
            if reviewer in self.constraint[paper]:
                raise ValueError(
                    f"Duplicate constraint pair in {constraints_file} at line "
                    f"{line_number}: ({self.paper_id(paper)}, {self.reviewer_id(reviewer)})"
                )
            constraint = _integer(
                constraint_value, "constraint", constraints_file, line_number
            )
            if constraint not in (-1, 1):
                raise ValueError(
                    f"Constraint must be -1 or 1 in {constraints_file} at line "
                    f"{line_number}; got {constraint}"
                )
            if constraint == 1 and reviewer not in self.s[paper]:
                raise ValueError(
                    f"Forced assignment ({self.paper_id(paper)}, "
                    f"{self.reviewer_id(reviewer)}) in {constraints_file} at line "
                    f"{line_number} is missing from similarity_scores.csv"
                )
            self.constraint[paper][reviewer] = constraint

    def _adjust_reviewer_loads_for_forced_assignments(self):
        forced_assignments_by_reviewer = [0 for _ in range(self.nr)]
        for paper_constraints in self.constraint:
            for reviewer, constraint in paper_constraints.items():
                if constraint == 1:
                    forced_assignments_by_reviewer[reviewer] += 1
        for reviewer, forced_assignments in enumerate(forced_assignments_by_reviewer):
            self.ellr[reviewer] = max(self.ellr[reviewer], forced_assignments)

    def constraint_for(self, paper, reviewer):
        return self.constraint[paper].get(reviewer, 0)

    def paper_id(self, internal_paper_id):
        return self.paper_internal_to_external[internal_paper_id]

    def reviewer_id(self, internal_reviewer_id):
        return self.reviewer_internal_to_external[internal_reviewer_id]

    def external_matching_pairs(self, matching_pairs):
        return [
            [self.paper_id(paper), self.reviewer_id(reviewer)]
            for paper, reviewer in matching_pairs
        ]

    def external_probability_pairs(self, probability_pairs):
        return [
            [self.paper_id(paper), self.reviewer_id(reviewer), probability]
            for paper, reviewer, probability in probability_pairs
        ]
