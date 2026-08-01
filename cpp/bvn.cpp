#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr int kProbabilityScale = 10000000;

struct Edge {
    int from = 0;
    int to = 0;
    int next = 0;
    int flow = 0;
    bool visited = false;
};

struct InstitutionLoad {
    int institution = 0;
    int next = 0;
    int load = 0;
    bool visited = false;
};

class BvnSampler {
  public:
    BvnSampler(int reviewer_count, int paper_count, bool optimize_sampling)
        : reviewer_count_(reviewer_count), paper_count_(paper_count),
          vertex_count_(reviewer_count + paper_count),
          optimize_sampling_(optimize_sampling), heads_(vertex_count_ + 1),
          vertex_loads_(vertex_count_ + 1), vertex_visited_(vertex_count_ + 1),
          reviewer_regions_(reviewer_count_ + 1),
          reviewer_institutions_(reviewer_count_ + 1),
          paper_institution_heads_(vertex_count_ + 1),
          coauthors_(reviewer_count_ + 1),
          violated_papers_(reviewer_count_ + 1),
          fractional_reviewers_by_paper_(paper_count_ + 1),
          random_engine_(static_cast<unsigned int>(
              std::chrono::high_resolution_clock::now().time_since_epoch().count())) {
        // Edge indices start at 2 so paired reverse edges can use index ^ 1.
        edges_.resize(2);
        institution_loads_.resize(1);
        edge_stack_.push_back(0);
    }

    void SetReviewerInfo(int reviewer, int region, int institution) {
        reviewer_regions_[reviewer] = region;
        reviewer_institutions_[reviewer] = institution;
    }

    void AddCoauthorPair(int first, int second) {
        coauthors_[first].insert(second);
        coauthors_[second].insert(first);
    }

    void AddAssignmentEdge(int reviewer_zero_based, int paper_vertex_zero_based,
                           double probability) {
        const int reviewer = reviewer_zero_based + 1;
        const int paper_vertex = paper_vertex_zero_based + 1;
        const int flow = static_cast<int>(
            std::llround(probability * static_cast<double>(kProbabilityScale)));

        vertex_loads_[reviewer] += flow;
        vertex_loads_[paper_vertex] -= flow;
        if (flow == 0) {
            return;
        }

        AddDirectedEdge(reviewer, paper_vertex, flow);
        AddDirectedEdge(paper_vertex, reviewer, kProbabilityScale - flow);
        AddInstitutionLoad(paper_vertex, reviewer_institutions_[reviewer], flow);
        RemovePairIfIntegral(static_cast<int>(edges_.size()) - 1);

        if (flow != kProbabilityScale) {
            fractional_reviewers_by_paper_[paper_vertex - reviewer_count_].push_back(
                reviewer);
        }
    }

    bool Round() {
        return optimize_sampling_ ? RoundWithHeuristics() : RoundBaseline();
    }

    void PrintMatching() const {
        for (std::size_t edge_index = 2; edge_index < edges_.size(); ++edge_index) {
            const Edge &edge = edges_[edge_index];
            if (edge.from < edge.to && edge.flow == kProbabilityScale) {
                std::cout << edge.from - 1 << ' ' << edge.to - 1 << '\n';
            }
        }
    }

  private:
    static int FloorMultiple(int value) {
        if (value >= 0) {
            return value / kProbabilityScale * kProbabilityScale;
        }
        return -((-value + kProbabilityScale - 1) / kProbabilityScale) *
               kProbabilityScale;
    }

    static int CeilMultiple(int value) { return -FloorMultiple(-value); }

    static bool IsIntegral(int value) { return value % kProbabilityScale == 0; }

    bool IsReviewer(int vertex) const { return vertex <= reviewer_count_; }

    void AddDirectedEdge(int from, int to, int flow) {
        edges_.push_back({from, to, heads_[from], flow, false});
        heads_[from] = static_cast<int>(edges_.size()) - 1;
        ++active_directed_edges_;
    }

    void RemoveDirectedEdge(int edge_index) {
        --active_directed_edges_;
        const int from = edges_[edge_index].from;
        if (heads_[from] == edge_index) {
            heads_[from] = edges_[edge_index].next;
            return;
        }

        int previous = heads_[from];
        while (previous != 0 && edges_[previous].next != edge_index) {
            previous = edges_[previous].next;
        }
        if (previous != 0) {
            edges_[previous].next = edges_[edge_index].next;
        }
    }

    void RemovePairIfIntegral(int edge_index) {
        if (edges_[edge_index].flow == 0 ||
            edges_[edge_index].flow == kProbabilityScale) {
            RemoveDirectedEdge(edge_index);
            RemoveDirectedEdge(edge_index ^ 1);
        }
    }

    int FindInstitutionLoad(int paper_vertex, int institution) const {
        for (int entry = paper_institution_heads_[paper_vertex]; entry != 0;
             entry = institution_loads_[entry].next) {
            if (institution_loads_[entry].institution == institution) {
                return entry;
            }
        }
        return 0;
    }

    void AddInstitutionLoad(int paper_vertex, int institution, int amount) {
        const int existing = FindInstitutionLoad(paper_vertex, institution);
        if (existing != 0) {
            institution_loads_[existing].load += amount;
            return;
        }

        institution_loads_.push_back(
            {institution, paper_institution_heads_[paper_vertex], amount, false});
        paper_institution_heads_[paper_vertex] =
            static_cast<int>(institution_loads_.size()) - 1;
    }

    void UpdateEdge(int edge_index, int amount) {
        Edge &edge = edges_[edge_index];
        Edge &reverse = edges_[edge_index ^ 1];
        edge.flow -= amount;
        reverse.flow += amount;
        vertex_loads_[edge.from] -= amount;
        vertex_loads_[edge.to] += amount;

        if (!IsReviewer(edge.to)) {
            AddInstitutionLoad(edge.to, reviewer_institutions_[edge.from], -amount);
        } else {
            AddInstitutionLoad(edge.from, reviewer_institutions_[edge.to], amount);
        }
        RemovePairIfIntegral(edge_index);
    }

    int ReviewerForEdge(int edge_index) const {
        const Edge &edge = edges_[edge_index];
        return IsReviewer(edge.from) ? edge.from : edge.to;
    }

    int PaperForEdge(int edge_index) const {
        const Edge &edge = edges_[edge_index];
        return (IsReviewer(edge.from) ? edge.to : edge.from) - reviewer_count_;
    }

    int EdgePriority(int edge_index) const {
        if (!optimize_sampling_) {
            return 0;
        }

        const int reviewer = ReviewerForEdge(edge_index);
        const int paper = PaperForEdge(edge_index);
        const int already_violated =
            violated_papers_[reviewer].count(paper) == 0 ? 0 : 1;
        if (edge_stack_.size() == 1) {
            return already_violated;
        }

        const int previous_reviewer = ReviewerForEdge(edge_stack_.back());
        if (reviewer == previous_reviewer) {
            return already_violated;
        }

        int priority = 1 - already_violated;
        if (coauthors_[reviewer].count(previous_reviewer) != 0) {
            priority += 4;
        }
        if (reviewer_regions_[reviewer] == reviewer_regions_[previous_reviewer]) {
            priority += 2;
        }
        return priority;
    }

    int SelectEdge(int vertex, int required_institution) const {
        if (IsReviewer(vertex)) {
            return SelectFromAdjacency(vertex, 0);
        }

        if (required_institution == 0) {
            for (int entry = paper_institution_heads_[vertex]; entry != 0;
                 entry = institution_loads_[entry].next) {
                if (!IsIntegral(institution_loads_[entry].load)) {
                    const int edge =
                        SelectEdge(vertex, institution_loads_[entry].institution);
                    if (edge != 0) {
                        return edge;
                    }
                }
            }
            return 0;
        }
        return SelectFromAdjacency(vertex, required_institution);
    }

    int SelectFromAdjacency(int vertex, int required_institution) const {
        int selected = 0;
        int selected_priority = 0;
        for (int edge = heads_[vertex]; edge != 0; edge = edges_[edge].next) {
            if (edges_[edge].visited) {
                continue;
            }
            if (required_institution != 0 &&
                reviewer_institutions_[edges_[edge].to] != required_institution) {
                continue;
            }
            if (!optimize_sampling_) {
                return edge;
            }

            const int priority = EdgePriority(edge);
            if (selected == 0 || priority > selected_priority) {
                selected = edge;
                selected_priority = priority;
            }
        }
        return selected;
    }

    bool Walk(int vertex, int incoming_edge, bool finding_path) {
        if (optimize_sampling_) {
            visited_vertex_stack_.push_back(vertex);
        }
        if (incoming_edge != 0) {
            edge_stack_.push_back(incoming_edge);
        }

        bool found = false;
        int selected_edge = 0;
        int incoming_institution_entry = 0;
        int outgoing_institution_entry = 0;

        if (IsReviewer(vertex)) {
            if (vertex_visited_[vertex]) {
                forward_limit_ = backward_limit_ = kProbabilityScale;
                stack_bottom_ = 0;
                for (std::size_t index = 1; index < edge_stack_.size(); ++index) {
                    if (edges_[edge_stack_[index]].from == vertex) {
                        stack_bottom_ = static_cast<int>(index);
                        break;
                    }
                }
                path_terminal_found_ = false;
                return true;
            }

            if (incoming_edge != 0 && finding_path &&
                !IsIntegral(vertex_loads_[vertex])) {
                forward_limit_ =
                    CeilMultiple(vertex_loads_[vertex]) - vertex_loads_[vertex];
                backward_limit_ =
                    vertex_loads_[vertex] - FloorMultiple(vertex_loads_[vertex]);
                stack_bottom_ = 1;
                path_terminal_found_ = true;
                return true;
            }

            vertex_visited_[vertex] = true;
            selected_edge = SelectEdge(vertex, 0);
            if (selected_edge == 0) {
                forward_limit_ = backward_limit_ = 0;
                return false;
            }

            edges_[selected_edge].visited = true;
            edges_[selected_edge ^ 1].visited = true;
            found = Walk(edges_[selected_edge].to, selected_edge, finding_path);
            edges_[selected_edge].visited = false;
            edges_[selected_edge ^ 1].visited = false;
            forward_limit_ =
                std::min(forward_limit_, edges_[selected_edge].flow);
            backward_limit_ =
                std::min(backward_limit_, edges_[selected_edge ^ 1].flow);
        } else {
            if (incoming_edge != 0) {
                incoming_institution_entry = FindInstitutionLoad(
                    vertex, reviewer_institutions_[edges_[incoming_edge].from]);
            }

            if (institution_loads_[incoming_institution_entry].visited) {
                forward_limit_ = backward_limit_ = kProbabilityScale;
                stack_bottom_ = 0;
                for (std::size_t index = 1; index < edge_stack_.size(); ++index) {
                    const Edge &edge = edges_[edge_stack_[index]];
                    if (edge.from == vertex &&
                        reviewer_institutions_[edge.to] ==
                            institution_loads_[incoming_institution_entry]
                                .institution) {
                        stack_bottom_ = static_cast<int>(index);
                        break;
                    }
                }
                path_terminal_found_ = false;
                return true;
            }

            if (vertex_visited_[vertex] &&
                !IsIntegral(institution_loads_[incoming_institution_entry].load)) {
                forward_limit_ = institution_loads_[incoming_institution_entry].load -
                                 FloorMultiple(
                                     institution_loads_[incoming_institution_entry].load);
                backward_limit_ =
                    CeilMultiple(institution_loads_[incoming_institution_entry].load) -
                    institution_loads_[incoming_institution_entry].load;
                stack_bottom_ = 0;

                int first_fractional_entry = 0;
                for (std::size_t index = 1; index < edge_stack_.size(); ++index) {
                    const Edge &edge = edges_[edge_stack_[index]];
                    if (edge.from != vertex) {
                        continue;
                    }
                    first_fractional_entry = FindInstitutionLoad(
                        vertex, reviewer_institutions_[edge.to]);
                    if (!IsIntegral(
                            institution_loads_[first_fractional_entry].load)) {
                        stack_bottom_ = static_cast<int>(index);
                        break;
                    }
                }

                forward_limit_ = std::min(
                    forward_limit_,
                    CeilMultiple(institution_loads_[first_fractional_entry].load) -
                        institution_loads_[first_fractional_entry].load);
                backward_limit_ = std::min(
                    backward_limit_,
                    institution_loads_[first_fractional_entry].load -
                        FloorMultiple(institution_loads_[first_fractional_entry].load));
                path_terminal_found_ = false;
                return true;
            }

            if (incoming_edge != 0 && finding_path &&
                !IsIntegral(vertex_loads_[vertex]) &&
                !IsIntegral(institution_loads_[incoming_institution_entry].load)) {
                forward_limit_ =
                    CeilMultiple(vertex_loads_[vertex]) - vertex_loads_[vertex];
                backward_limit_ =
                    vertex_loads_[vertex] - FloorMultiple(vertex_loads_[vertex]);
                forward_limit_ = std::min(
                    forward_limit_,
                    institution_loads_[incoming_institution_entry].load -
                        FloorMultiple(
                            institution_loads_[incoming_institution_entry].load));
                backward_limit_ = std::min(
                    backward_limit_,
                    CeilMultiple(institution_loads_[incoming_institution_entry].load) -
                        institution_loads_[incoming_institution_entry].load);
                stack_bottom_ = 1;
                path_terminal_found_ = true;
                return true;
            }

            if (IsIntegral(institution_loads_[incoming_institution_entry].load)) {
                selected_edge = SelectEdge(
                    vertex,
                    institution_loads_[incoming_institution_entry].institution);
            } else {
                selected_edge = SelectEdge(vertex, 0);
            }
            if (selected_edge == 0) {
                forward_limit_ = backward_limit_ = 0;
                return false;
            }

            outgoing_institution_entry = FindInstitutionLoad(
                vertex, reviewer_institutions_[edges_[selected_edge].to]);
            institution_loads_[outgoing_institution_entry].visited = true;
            edges_[selected_edge].visited = true;
            edges_[selected_edge ^ 1].visited = true;
            if (!IsIntegral(institution_loads_[outgoing_institution_entry].load)) {
                vertex_visited_[vertex] = true;
            }

            found = Walk(edges_[selected_edge].to, selected_edge, finding_path);

            institution_loads_[outgoing_institution_entry].visited = false;
            edges_[selected_edge].visited = false;
            edges_[selected_edge ^ 1].visited = false;
            forward_limit_ =
                std::min(forward_limit_, edges_[selected_edge].flow);
            backward_limit_ =
                std::min(backward_limit_, edges_[selected_edge ^ 1].flow);
        }

        if (selected_edge == edge_stack_[stack_bottom_] &&
            forward_limit_ + backward_limit_ != 0) {
            const bool starts_path =
                optimize_sampling_ ? path_terminal_found_
                                   : (incoming_edge == 0 && finding_path);
            if (starts_path) {
                forward_limit_ = std::min(
                    forward_limit_,
                    vertex_loads_[vertex] - FloorMultiple(vertex_loads_[vertex]));
                backward_limit_ = std::min(
                    backward_limit_,
                    CeilMultiple(vertex_loads_[vertex]) - vertex_loads_[vertex]);
                if (!IsReviewer(vertex)) {
                    const int entry = FindInstitutionLoad(
                        vertex, reviewer_institutions_[edges_[selected_edge].to]);
                    forward_limit_ = std::min(
                        forward_limit_,
                        CeilMultiple(institution_loads_[entry].load) -
                            institution_loads_[entry].load);
                    backward_limit_ = std::min(
                        backward_limit_,
                        institution_loads_[entry].load -
                            FloorMultiple(institution_loads_[entry].load));
                }
            }

            std::uniform_int_distribution<int> choose_direction(
                1, forward_limit_ + backward_limit_);
            const bool update_forward =
                choose_direction(random_engine_) <= backward_limit_;
            const int amount = update_forward ? forward_limit_ : -backward_limit_;
            for (std::size_t index = stack_bottom_; index < edge_stack_.size();
                 ++index) {
                UpdateEdge(edge_stack_[index], amount);
            }
            forward_limit_ = backward_limit_ = 0;
        }

        if (!IsReviewer(vertex) &&
            incoming_institution_entry != outgoing_institution_entry) {
            forward_limit_ = std::min(
                forward_limit_,
                CeilMultiple(institution_loads_[outgoing_institution_entry].load) -
                    institution_loads_[outgoing_institution_entry].load);
            backward_limit_ = std::min(
                backward_limit_,
                institution_loads_[outgoing_institution_entry].load -
                    FloorMultiple(
                        institution_loads_[outgoing_institution_entry].load));
            forward_limit_ = std::min(
                forward_limit_,
                institution_loads_[incoming_institution_entry].load -
                    FloorMultiple(
                        institution_loads_[incoming_institution_entry].load));
            backward_limit_ = std::min(
                backward_limit_,
                CeilMultiple(institution_loads_[incoming_institution_entry].load) -
                    institution_loads_[incoming_institution_entry].load);
        }
        return found;
    }

    void ResetStacks() {
        edge_stack_.resize(1);
        stack_bottom_ = 0;
        forward_limit_ = backward_limit_ = 0;
        path_terminal_found_ = false;
    }

    void ResetOptimizedSearch() {
        for (const int vertex : visited_vertex_stack_) {
            vertex_visited_[vertex] = false;
        }
        visited_vertex_stack_.clear();
        ResetStacks();
    }

    bool RoundBaseline() {
        while (active_directed_edges_ != 0) {
            bool made_progress = false;
            std::fill(vertex_visited_.begin(), vertex_visited_.end(), false);
            for (int vertex = 1; vertex <= vertex_count_; ++vertex) {
                if (IsIntegral(vertex_loads_[vertex])) {
                    continue;
                }
                ResetStacks();
                if (Walk(vertex, 0, true)) {
                    made_progress = true;
                    break;
                }
            }

            if (active_directed_edges_ == 0) {
                break;
            }
            std::fill(vertex_visited_.begin(), vertex_visited_.end(), false);
            for (int vertex = 1; vertex <= vertex_count_; ++vertex) {
                ResetStacks();
                if (Walk(vertex, 0, false)) {
                    made_progress = true;
                    break;
                }
            }
            if (!made_progress) {
                return false;
            }
        }
        return true;
    }

    void InitializeHeuristics(std::vector<int> &reviewer_order) {
        for (int paper = 1; paper <= paper_count_; ++paper) {
            const std::vector<int> &reviewers =
                fractional_reviewers_by_paper_[paper];
            for (std::size_t first = 0; first < reviewers.size(); ++first) {
                for (std::size_t second = first + 1; second < reviewers.size();
                     ++second) {
                    const int first_reviewer = reviewers[first];
                    const int second_reviewer = reviewers[second];
                    if (coauthors_[first_reviewer].count(second_reviewer) != 0) {
                        violated_papers_[first_reviewer].insert(paper);
                        violated_papers_[second_reviewer].insert(paper);
                    }
                }
            }
        }

        reviewer_order.resize(reviewer_count_);
        for (int reviewer = 1; reviewer <= reviewer_count_; ++reviewer) {
            reviewer_order[reviewer - 1] = reviewer;
        }
        std::sort(reviewer_order.begin(), reviewer_order.end(),
                  [this](int first, int second) {
                      return violated_papers_[first].size() >
                             violated_papers_[second].size();
                  });
    }

    bool RoundWithHeuristics() {
        std::vector<int> reviewer_order;
        InitializeHeuristics(reviewer_order);

        std::size_t next_reviewer = 0;
        while (next_reviewer < reviewer_order.size()) {
            while (next_reviewer < reviewer_order.size() &&
                   IsIntegral(vertex_loads_[reviewer_order[next_reviewer]])) {
                ++next_reviewer;
            }
            if (next_reviewer == reviewer_order.size()) {
                break;
            }
            ResetOptimizedSearch();
            if (!Walk(reviewer_order[next_reviewer], 0, true)) {
                ++next_reviewer;
            }
        }

        int next_paper = 1;
        while (next_paper <= paper_count_) {
            while (next_paper <= paper_count_ &&
                   IsIntegral(vertex_loads_[reviewer_count_ + next_paper])) {
                ++next_paper;
            }
            if (next_paper > paper_count_) {
                break;
            }
            ResetOptimizedSearch();
            if (!Walk(reviewer_count_ + next_paper, 0, true)) {
                ++next_paper;
            }
        }

        next_reviewer = 0;
        while (active_directed_edges_ != 0) {
            if (next_reviewer == reviewer_order.size()) {
                return false;
            }
            ResetOptimizedSearch();
            if (!Walk(reviewer_order[next_reviewer], 0, false)) {
                ++next_reviewer;
            }
        }
        return true;
    }

    int reviewer_count_;
    int paper_count_;
    int vertex_count_;
    bool optimize_sampling_;
    int active_directed_edges_ = 0;
    int stack_bottom_ = 0;
    int forward_limit_ = 0;
    int backward_limit_ = 0;
    bool path_terminal_found_ = false;

    std::vector<int> heads_;
    std::vector<Edge> edges_;
    std::vector<int> vertex_loads_;
    std::vector<bool> vertex_visited_;
    std::vector<int> reviewer_regions_;
    std::vector<int> reviewer_institutions_;
    std::vector<int> paper_institution_heads_;
    std::vector<InstitutionLoad> institution_loads_;
    std::vector<std::unordered_set<int>> coauthors_;
    std::vector<std::unordered_set<int>> violated_papers_;
    std::vector<std::vector<int>> fractional_reviewers_by_paper_;
    std::vector<int> edge_stack_;
    std::vector<int> visited_vertex_stack_;
    std::mt19937 random_engine_;
};

bool ParseOptimizeSampling(int argc, char **argv, bool &optimize_sampling) {
    optimize_sampling = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--optimize-sampling") {
            optimize_sampling = true;
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    bool optimize_sampling = false;
    if (!ParseOptimizeSampling(argc, argv, optimize_sampling)) {
        return 2;
    }

    int reviewer_count = 0;
    int paper_count = 0;
    if (!(std::cin >> reviewer_count >> paper_count) || reviewer_count <= 0 ||
        paper_count <= 0) {
        std::cerr << "Expected positive reviewer and paper counts\n";
        return 1;
    }

    BvnSampler sampler(reviewer_count, paper_count, optimize_sampling);
    for (int reviewer = 1; reviewer <= reviewer_count; ++reviewer) {
        int region = 0;
        int institution = 0;
        if (!(std::cin >> region >> institution) || region <= 0 ||
            institution <= 0) {
            std::cerr << "Invalid reviewer metadata\n";
            return 1;
        }
        sampler.SetReviewerInfo(reviewer, region, institution);
    }

    int coauthor_pair_count = 0;
    if (!(std::cin >> coauthor_pair_count) || coauthor_pair_count < 0) {
        std::cerr << "Invalid coauthor-pair count\n";
        return 1;
    }
    for (int pair = 0; pair < coauthor_pair_count; ++pair) {
        int first = 0;
        int second = 0;
        if (!(std::cin >> first >> second) || first <= 0 ||
            first > reviewer_count || second <= 0 || second > reviewer_count ||
            first == second) {
            std::cerr << "Invalid coauthor pair\n";
            return 1;
        }
        sampler.AddCoauthorPair(first, second);
    }

    int reviewer = 0;
    int paper_vertex = 0;
    double probability = 0.0;
    while (std::cin >> reviewer >> paper_vertex >> probability) {
        if (reviewer < 0 || reviewer >= reviewer_count ||
            paper_vertex < reviewer_count ||
            paper_vertex >= reviewer_count + paper_count ||
            !std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
            std::cerr << "Invalid assignment edge\n";
            return 1;
        }
        sampler.AddAssignmentEdge(reviewer, paper_vertex, probability);
    }

    if (!std::cin.eof()) {
        std::cerr << "Malformed assignment edge\n";
        return 1;
    }
    if (!sampler.Round()) {
        std::cerr << "Unable to round the fractional assignment\n";
        return 1;
    }
    sampler.PrintMatching();
    return 0;
}
