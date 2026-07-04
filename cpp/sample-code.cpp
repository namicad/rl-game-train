#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct Model {
  std::vector<float> weights;
  std::vector<float> bias;
  int input_dim = 0;
  int action_dim = 0;
};

constexpr int MAX_TURN = 200;         // maximum turn (days)
constexpr int START_GOLD = 500;       // initial gold
constexpr int START_WARRIORS = 3;     // initial warriors
constexpr int MOVE_COST = 10;         // move cost
constexpr int TRAIN_COST = 120;       // train cost
constexpr int WORK_INCOME = 15;       // income per warrior
constexpr int UPKEEP_PER_WARRIOR = 2; // upkeep per warrior
constexpr int HQ_MAX_LEVEL = 5;       // HQ max level
constexpr int BASE_MAX_LEVEL = 3;     // base max level
constexpr int HQ_HEAL_COST = 1000;    // HQ fix cost
constexpr int BASE_HEAL_COST = 500;   // base fix cost

struct HqLevelEntry {
  int upgrade_cost;
  int warrior_hp;
  int hp;
  int turret;
  int train_cap;
  int work_cap;
};

struct BaseLevelEntry {
  int cost;
  int hp;
  int turret;
  int work_cap;
};

constexpr HqLevelEntry HQ_LEVELS[HQ_MAX_LEVEL + 1] = {
    {0, 0, 0, 0, 0, 0},     {0, 4, 10, 1, 1, 1},    {600, 5, 15, 2, 1, 2},
    {1200, 6, 20, 2, 2, 3}, {2400, 7, 25, 3, 2, 4}, {3600, 8, 30, 3, 3, 5},
};
constexpr BaseLevelEntry BASE_LEVELS[BASE_MAX_LEVEL + 1] = {
    {0, 0, 0, 0},
    {300, 6, 1, 1},
    {600, 12, 1, 2},
    {1000, 18, 2, 3},
};

enum class Side : int { LEFT = 0, RIGHT = 1 };
enum class BType : int { HQ, BASE };
enum class WState : int { STATIONARY, MOVING };

inline Side opposite(Side s) {
  return s == Side::LEFT ? Side::RIGHT : Side::LEFT;
}
inline char side_char(Side s) { return s == Side::LEFT ? 'A' : 'B'; }
inline Side parse_side_char(char c) {
  return c == 'A' ? Side::LEFT : Side::RIGHT;
}

struct WarriorId {
  Side side = Side::LEFT;
  int num = 0;
  bool operator==(const WarriorId &o) const {
    return side == o.side && num == o.num;
  }
};

struct Warrior {
  WarriorId id;
  int region = 0;
  int hp = 0;
  WState state = WState::STATIONARY;
  int target = 0;
};

struct Building {
  int region = 0;
  Side side = Side::LEFT;
  BType type = BType::HQ;
  int level = 1;
  int hp = 10;

  int current_hp() const {
    return type == BType::HQ ? HQ_LEVELS[level].hp : BASE_LEVELS[level].hp;
  }
  int work_cap() const {
    return type == BType::HQ ? HQ_LEVELS[level].work_cap
                             : BASE_LEVELS[level].work_cap;
  }
};

struct GameMap {
  int N = 0, K = 0;
  std::vector<long long> x, y;
  std::vector<int> strongholds;
  std::vector<std::vector<int>> adj;

  Side my_side = Side::LEFT;
  int my_hq = 0;
  int opp_hq = 0;
};

struct GameState {
  int gold = START_GOLD; // current gold
  int my_countdown = 5;  // my remaining countdowns
  int opp_countdown = 5; // opponent's remaining countdowns
  std::vector<Warrior> warriors;
  std::vector<Building> buildings;
};

struct Actions {
  int train_n = 0;
  std::vector<std::pair<WarriorId, int>> moves;
  std::vector<int> upgrades;
};

static std::string readln() {
  std::string s;
  if (!std::getline(std::cin, s))
    std::exit(0);
  return s;
}

static std::vector<std::string> tokens(const std::string &s) {
  std::vector<std::string> out;
  std::istringstream is(s);
  for (std::string t; is >> t;)
    out.push_back(t);
  return out;
}

static WarriorId parse_warrior(const std::string &tok) {
  assert(!tok.empty() && (tok[0] == 'A' || tok[0] == 'B'));
  WarriorId id;
  id.side = parse_side_char(tok[0]);
  id.num = std::stoi(tok.substr(1));
  return id;
}

static std::string format_warrior(WarriorId id) {
  std::string s;
  s.push_back(side_char(id.side));
  s += std::to_string(id.num);
  return s;
}

static int hq_of(const GameMap &M, Side s) {
  return (s == Side::LEFT) ? 0 : M.N - 1;
}

static Building make_base(int region, Side s) {
  return Building{region, s, BType::BASE, 1, BASE_LEVELS[1].hp};
}

static void apply_upgrade(Building &b) {
  b.level += 1;
  b.hp = b.current_hp();
}

static int upgrade_cost(const Building &b) {
  if (b.type == BType::HQ)
    return HQ_LEVELS[b.level + 1].upgrade_cost;
  else
    return BASE_LEVELS[b.level + 1].cost;
}

static int max_level(const Building &b) {
  return b.type == BType::HQ ? HQ_MAX_LEVEL : BASE_MAX_LEVEL;
}

static void parse_init(GameMap &M, GameState &S) {
  {
    auto t = tokens(readln());
    assert(t.size() >= 2 && t[0] == "READY");
    M.my_side = (t[1] == "LEFT") ? Side::LEFT : Side::RIGHT;
  }
  {
    auto t = tokens(readln());
    M.N = std::stoi(t.at(0));
    M.K = std::stoi(t.at(1));
  }
  M.x.assign(M.N, 0);
  M.y.assign(M.N, 0);
  {
    auto t = tokens(readln()); // x_0 x_1 ... x_{N-1}
    for (int i = 0; i < M.N; ++i)
      M.x[i] = std::stoll(t.at(i));
  }
  {
    auto t = tokens(readln()); // y_0 y_1 ... y_{N-1}
    for (int i = 0; i < M.N; ++i)
      M.y[i] = std::stoll(t.at(i));
  }
  {
    auto t = tokens(readln()); // K strongholds
    M.strongholds.clear();
    M.strongholds.reserve(t.size());
    for (const auto &s : t)
      M.strongholds.push_back(std::stoi(s));
    std::sort(M.strongholds.begin(), M.strongholds.end());
  }
  M.adj.assign(M.N, {});
  for (int r = 0; r < M.N; ++r) {
    auto t = tokens(readln()); // deg n_1 n_2 ...
    int deg = std::stoi(t.at(0));
    auto &nb = M.adj[r];
    nb.reserve(deg);
    for (int j = 0; j < deg; ++j)
      nb.push_back(std::stoi(t.at(1 + j)));
    std::sort(nb.begin(), nb.end());
  }

  M.my_hq = hq_of(M, M.my_side);
  M.opp_hq = hq_of(M, opposite(M.my_side));

  S = GameState{};
  S.gold = START_GOLD;
  Side opp = opposite(M.my_side);
  for (int sfx = 1; sfx <= START_WARRIORS; ++sfx) {
    Warrior my_warrior;
    my_warrior.id = WarriorId{M.my_side, sfx};
    my_warrior.region = M.my_hq;
    my_warrior.hp = HQ_LEVELS[1].warrior_hp;
    S.warriors.push_back(my_warrior);

    Warrior opp_warrior;
    opp_warrior.id = WarriorId{opp, sfx};
    opp_warrior.region = M.opp_hq;
    opp_warrior.hp = HQ_LEVELS[1].warrior_hp;
    S.warriors.push_back(opp_warrior);
  }
  Building left_hq;
  left_hq.region = hq_of(M, Side::LEFT);
  left_hq.side = Side::LEFT;
  left_hq.type = BType::HQ;
  left_hq.level = 1;
  left_hq.hp = HQ_LEVELS[1].hp;
  S.buildings.push_back(left_hq);
  S.buildings.push_back(Building{hq_of(M, Side::RIGHT), Side::RIGHT, BType::HQ,
                                 1, HQ_LEVELS[1].hp});

  std::cout << "OK" << std::endl;
}

static bool read_turn_start(int &turn_index) {
  std::string line = readln();
  if (line == "FINISH")
    return false;
  auto t = tokens(line);
  assert(!t.empty() && t[0] == "START");
  turn_index = std::stoi(t.at(2));
  return true;
}

static Building *find_building(GameState &S, int region) {
  for (auto &b : S.buildings)
    if (b.region == region)
      return &b;
  return nullptr;
}

static Warrior *find_warrior(GameState &S, WarriorId id) {
  for (auto &w : S.warriors)
    if (w.id == id)
      return &w;
  return nullptr;
}

static void read_turn_result(GameState &S, const GameMap &M,
                             const Actions &submitted) {
  for (int region : submitted.upgrades) {
    Building *b = find_building(S, region);
    if (b == nullptr) {
      S.gold -= BASE_LEVELS[1].cost;
      S.buildings.push_back(make_base(region, M.my_side));
    } else {
      if (b->level >= max_level(*b)) {
        int cost = (b->type == BType::HQ) ? HQ_HEAL_COST : BASE_HEAL_COST;
        S.gold -= cost;
        b->hp = b->current_hp();
      } else {
        S.gold -= upgrade_cost(*b);
        apply_upgrade(*b);
      }
    }
  }

  for (const auto &entry : submitted.moves) {
    const WarriorId &id = entry.first;
    const int target = entry.second;
    Building *b = find_building(S, target);
    int cost = (b != nullptr && b->side == M.my_side) ? 0 : MOVE_COST;
    S.gold -= cost;
    if (Warrior *w = find_warrior(S, id)) {
      w->state = WState::MOVING;
      w->target = target;
    }
  }

  S.gold -= TRAIN_COST * submitted.train_n;

  {
    std::string line = readln();
    if (line == "FINISH")
      std::exit(0);
    auto t = tokens(line);
    assert(!t.empty() && t[0] == "TURN");
  }
  {
    auto t = tokens(readln());
    S.my_countdown = std::stoi(t.at(2));
    S.opp_countdown = std::stoi(t.at(4));
  }
  // UPGRADE
  {
    auto t = tokens(readln()); // "UPGRADE N"
    int n = std::stoi(t.at(1));
    for (int i = 0; i < n; ++i) {
      auto r = tokens(readln()); // "<A|B> <region>"
      Side s = parse_side_char(r.at(0)[0]);
      int region = std::stoi(r.at(1));
      Building *b = find_building(S, region);
      if (b == nullptr) {
        S.buildings.push_back(make_base(region, s));
      } else if (b->side != M.my_side) {
        if (b->level >= max_level(*b)) {
          b->hp = b->current_hp();
        } else {
          apply_upgrade(*b);
        }
      }
    }
  }
  // TRAIN
  {
    auto t = tokens(readln()); // "TRAIN N"
    int n = std::stoi(t.at(1));
    if (n > 0) {
      auto ids = tokens(readln());
      for (int i = 0; i < n; ++i) {
        WarriorId id = parse_warrior(ids.at(i));
        int hq_region = hq_of(M, id.side);
        Building *hq_b = find_building(S, hq_region);
        int hq_level = (hq_b != nullptr) ? hq_b->level : 1;
        Warrior trained_warrior;
        trained_warrior.id = id;
        trained_warrior.region = hq_region;
        trained_warrior.hp = HQ_LEVELS[hq_level].warrior_hp;
        S.warriors.push_back(trained_warrior);
      }
    }
  }
  // MOVE
  {
    auto t = tokens(readln()); // "MOVE N"
    int n = std::stoi(t.at(1));
    for (int i = 0; i < n; ++i) {
      auto r = tokens(readln());
      WarriorId id = parse_warrior(r.at(0));
      int region = std::stoi(r.at(1));
      if (Warrior *w = find_warrior(S, id)) {
        w->region = region;
        if (id.side == M.my_side && w->state == WState::MOVING &&
            w->region == w->target) {
          w->state = WState::STATIONARY;
        }
      }
    }
  }
  // DAMAGE
  {
    auto t = tokens(readln()); // "DAMAGE N"
    int n = std::stoi(t.at(1));
    for (int i = 0; i < n; ++i) {
      auto r = tokens(readln());
      WarriorId id = parse_warrior(r.at(1));
      int damage = std::stoi(r.at(2));
      if (Warrior *w = find_warrior(S, id))
        w->hp -= damage;
    }
    S.warriors.erase(std::remove_if(S.warriors.begin(), S.warriors.end(),
                                    [](const Warrior &w) { return w.hp <= 0; }),
                     S.warriors.end());
  }
  // SIEGE
  {
    auto t = tokens(readln()); // "SIEGE N"
    int n = std::stoi(t.at(1));
    for (int i = 0; i < n; ++i) {
      auto r = tokens(readln());
      int region = std::stoi(r.at(1));
      int damage = std::stoi(r.at(2));
      if (Building *b = find_building(S, region))
        b->hp -= damage;
    }
    S.buildings.erase(
        std::remove_if(S.buildings.begin(), S.buildings.end(),
                       [](const Building &b) { return b.hp <= 0; }),
        S.buildings.end());
  }
  (void)readln(); // "END"

  int income = 0;
  for (const auto &b : S.buildings) {
    if (b.side != M.my_side)
      continue;
    int count = 0;
    for (const auto &w : S.warriors) {
      if (w.id.side == M.my_side && w.region == b.region)
        ++count;
    }
    income += WORK_INCOME * std::min(count, b.work_cap());
  }
  S.gold += income;

  int alive = 0;
  for (const auto &w : S.warriors)
    if (w.id.side == M.my_side)
      ++alive;
  S.gold = std::max(0, S.gold - UPKEEP_PER_WARRIOR * alive);
}

struct Paths {
  std::vector<std::vector<double>> dist;
  std::vector<std::vector<int>> nxt;
};

static double euclid_ceil(const GameMap &M, int u, int v) {
  double dx = (double)(M.x[u] - M.x[v]);
  double dy = (double)(M.y[u] - M.y[v]);
  return std::ceil(std::sqrt(dx * dx + dy * dy));
}

static Paths calculate_paths(const GameMap &M) {
  const double INF = std::numeric_limits<double>::infinity();
  Paths P;
  P.dist.assign(M.N, std::vector<double>(M.N, INF));
  P.nxt.assign(M.N, std::vector<int>(M.N, -1));

  for (int i = 0; i < M.N; ++i) {
    P.dist[i][i] = 0.0;
    P.nxt[i][i] = i;
  }
  for (int u = 0; u < M.N; ++u) {
    for (int v : M.adj[u]) {
      double w = euclid_ceil(M, u, v);
      if (w < P.dist[u][v])
        P.dist[u][v] = w;
    }
  }

  for (int k = 0; k < M.N; ++k) {
    for (int u = 0; u < M.N; ++u) {
      if (P.dist[u][k] == INF)
        continue;
      for (int v = 0; v < M.N; ++v) {
        double cand = P.dist[u][k] + P.dist[k][v];
        if (cand < P.dist[u][v])
          P.dist[u][v] = cand;
      }
    }
  }

  for (int u = 0; u < M.N; ++u) {
    for (int v = 0; v < M.N; ++v) {
      if (u == v || P.dist[u][v] == INF)
        continue;
      double best_score = INF;
      for (int nb : M.adj[u]) {
        if (P.dist[nb][v] == INF)
          continue;
        double score = euclid_ceil(M, u, nb) + P.dist[nb][v];
        if (score < best_score) {
          best_score = score;
          P.nxt[u][v] = nb;
        }
      }
    }
  }
  return P;
}

// Returns the next step on the path from u to v.
// If the path is not reachable, returns -1.
static int next_step(const Paths &P, int u, int v) { return P.nxt[u][v]; }

// Returns the path from u to v.
// If the path is not reachable, returns an empty vector.
static std::vector<int> path(const Paths &P, int u, int v) {
  std::vector<int> out;
  if (P.nxt[u][v] == -1)
    return out;
  out.push_back(u);
  while (u != v) {
    u = P.nxt[u][v];
    out.push_back(u);
  }
  return out;
}

static void emit_command() { std::cout << "COMMAND\n"; }

static void emit_actions(const Actions &a) {
  for (const auto &entry : a.moves) {
    const WarriorId &id = entry.first;
    const int target = entry.second;
    std::cout << "MOVE " << format_warrior(id) << ' ' << target << '\n';
  }
  for (int r : a.upgrades) {
    std::cout << "UPGRADE " << r << '\n';
  }
  if (a.train_n > 0) {
    std::cout << "TRAIN " << a.train_n << '\n';
  }
}

static void emit_end() { std::cout << "END" << std::endl; }

//////////////////////////////////
//// WRITE YOUR STRATEGY HERE ////
//////////////////////////////////

// AUTO-RL-UPDATE:START
static const char* kLatestRlSummary = "epochs=600;final_loss=11.861469;eval_match_rate=0.633333";
// AUTO-RL-UPDATE:END




struct RlSummaryMeta {
  int epochs = 0;
  float final_loss = 0.0f;
  float eval_match_rate = 0.0f;
  bool valid = false;
};

RlSummaryMeta parse_rl_summary(const char* raw) {
  RlSummaryMeta meta;
  if (raw == nullptr) {
    return meta;
  }

  std::string text(raw);
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ';')) {
    const size_t eq = token.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = token.substr(0, eq);
    const std::string value = token.substr(eq + 1);
    if (key == "epochs") {
      meta.epochs = std::atoi(value.c_str());
    } else if (key == "final_loss") {
      meta.final_loss = std::atof(value.c_str());
    } else if (key == "eval_match_rate") {
      meta.eval_match_rate = std::atof(value.c_str());
    }
  }

  meta.valid = true;
  return meta;
}

const RlSummaryMeta &get_latest_rl_summary() {
  static const RlSummaryMeta summary = parse_rl_summary(kLatestRlSummary);
  return summary;
}

namespace {

constexpr int kRlInputDim = 16;
constexpr int kRlActionDim = 4;

Model load_rl_model(const char *path) {
  Model m;
  const char *file_path = path != nullptr ? path : "models/weights.txt";
  std::ifstream input(file_path);
  if (!input.is_open()) {
    m.weights = {1.0f};
    m.bias = {0.0f};
    m.input_dim = 1;
    m.action_dim = 1;
    return m;
  }

  std::vector<float> values;
  std::string line;
  std::getline(input, line);
  std::istringstream iss(line);
  float value = 0.0f;
  while (iss >> value) {
    values.push_back(value);
  }

  m.input_dim = kRlInputDim;
  m.action_dim = kRlActionDim;
  const int weight_count = m.input_dim * m.action_dim;
  const int bias_count = m.action_dim;

  if (values.size() >= static_cast<size_t>(weight_count + bias_count)) {
    m.weights.assign(values.begin(), values.begin() + weight_count);
    m.bias.assign(values.begin() + weight_count,
                  values.begin() + weight_count + bias_count);
  } else if (!values.empty()) {
    m.weights.assign(values.begin(), values.end());
    m.bias.assign(m.action_dim, 0.0f);
  } else {
    m.weights.assign(weight_count, 0.0f);
    m.bias.assign(bias_count, 0.0f);
  }

  return m;
}

std::vector<float> predict_rl_logits(const Model &m,
                                     const std::vector<float> &input) {
  std::vector<float> logits(m.action_dim, 0.0f);
  if (m.weights.empty() || m.action_dim <= 0 || m.input_dim <= 0) {
    return logits;
  }

  std::vector<float> normalized_input(input.begin(), input.end());
  float norm = 0.0f;
  for (float value : normalized_input) {
    norm += value * value;
  }
  norm = std::sqrt(norm);
  if (norm > 1e-8f) {
    for (float &value : normalized_input) {
      value /= norm;
    }
  }

  for (int action = 0; action < m.action_dim; ++action) {
    float score = 0.0f;
    const int base = action * m.input_dim;
    for (int feature = 0; feature < m.input_dim; ++feature) {
      const int index = base + feature;
      if (index >= static_cast<int>(m.weights.size())) {
        break;
      }
      const float value = feature < static_cast<int>(normalized_input.size()) ? normalized_input[feature] : 0.0f;
      score += value * m.weights[index];
    }
    if (action < static_cast<int>(m.bias.size())) {
      score += m.bias[action];
    }
    logits[action] = score;
  }
  return logits;
}

float predict_rl(const Model &m, const std::vector<float> &input) {
  const auto logits = predict_rl_logits(m, input);
  return logits.empty() ? 0.0f : logits.front();
}

} // namespace

namespace strategy {

struct AnalyzerOutput {
  int turn = 0;
  int gold = 0;
  int own_warriors = 0;
  int enemy_warriors = 0;
  int friendly_buildings = 0;
  int enemy_buildings = 0;
  bool opening_turn = false;
  bool can_train = false;
  bool can_upgrade = false;
  bool under_pressure = false;
  int pressure_score = 0;
  int my_hq_hp = 0;
  int enemy_hq_hp = 0;
};

struct EconomyCandidate {
  int train_n = 0;
  std::vector<int> upgrades;
  std::vector<std::pair<WarriorId, int>> moves;
  float score = 0.0f;
  std::string label;
};

struct AttackCandidate {
  std::vector<std::pair<WarriorId, int>> moves;
  float score = 0.0f;
  std::string label;
};

struct DefenseCandidate {
  std::vector<int> upgrades;
  std::vector<std::pair<WarriorId, int>> moves;
  float score = 0.0f;
  std::string label;
};

struct PlanBundle {
  EconomyCandidate economy;
  AttackCandidate attack;
  DefenseCandidate defense;
};

class Analyzer {
public:
  AnalyzerOutput analyze(const GameState &state, const GameMap &map,
                         const Paths &, int turn) const {
    AnalyzerOutput out;
    out.turn = turn;
    out.gold = state.gold;
    out.opening_turn = (turn == 1);

    for (const auto &warrior : state.warriors) {
      if (warrior.id.side == map.my_side) {
        ++out.own_warriors;
      } else {
        ++out.enemy_warriors;
      }
    }

    for (const auto &building : state.buildings) {
      if (building.side == map.my_side) {
        ++out.friendly_buildings;
      } else {
        ++out.enemy_buildings;
      }
    }

    out.can_train = out.gold >= TRAIN_COST;
    out.can_upgrade = out.gold >= 600 && out.friendly_buildings > 0;
    out.under_pressure = out.enemy_warriors > out.own_warriors || out.own_warriors <= 2;
    out.pressure_score = out.enemy_warriors - out.own_warriors;
    if (out.opening_turn) {
      out.pressure_score += 2;
    }

    for (const auto &building : state.buildings) {
      if (building.side == map.my_side && building.type == BType::HQ) {
        out.my_hq_hp = building.hp;
      }
      if (building.side != map.my_side && building.type == BType::HQ) {
        out.enemy_hq_hp = building.hp;
      }
    }
    return out;
  }
};

class EconomyPlanner {
public:
  std::vector<EconomyCandidate> plan(const AnalyzerOutput &summary,
                                     const GameState &state,
                                     const GameMap &map) const {
    std::vector<EconomyCandidate> candidates;

    EconomyCandidate idle;
    idle.label = "idle";
    idle.score = 70.0f + static_cast<float>(summary.friendly_buildings) * 2.0f;
    if (summary.my_hq_hp <= 12) {
      idle.score -= 12.0f;
    }
    candidates.push_back(idle);

    if (summary.can_train && summary.turn > 1) {
      EconomyCandidate train;
      train.train_n = 1;
      train.label = "train";
      train.score = 60.0f - static_cast<float>(summary.gold) / 1000.0f +
                    static_cast<float>(summary.turn) * 0.1f;
      if (summary.under_pressure) {
        train.score += 8.0f;
      }
      if (summary.enemy_hq_hp <= 15) {
        train.score += 3.0f;
      }
      candidates.push_back(train);
    }

    if (summary.can_train && summary.can_upgrade && summary.turn > 2) {
      EconomyCandidate train_and_upgrade;
      train_and_upgrade.train_n = 1;
      train_and_upgrade.upgrades.push_back(map.my_hq);
      train_and_upgrade.label = "train_upgrade";
      train_and_upgrade.score = 55.0f + static_cast<float>(summary.friendly_buildings) * 3.0f;
      if (summary.my_hq_hp <= 15) {
        train_and_upgrade.score += 6.0f;
      }
      if (summary.own_warriors >= 3) {
        train_and_upgrade.score += 2.0f;
      }
      candidates.push_back(train_and_upgrade);
    }

    if (summary.can_upgrade) {
      EconomyCandidate upgrade;
      upgrade.upgrades.push_back(map.my_hq);
      upgrade.label = "upgrade";
      upgrade.score = 45.0f - static_cast<float>(summary.gold) / 2000.0f +
                      static_cast<float>(summary.friendly_buildings) * 2.5f;
      if (summary.my_hq_hp <= 12) {
        upgrade.score += 10.0f;
      }
      if (summary.under_pressure) {
        upgrade.score -= 4.0f;
      }
      candidates.push_back(upgrade);
    }

    if (!state.warriors.empty() && summary.turn > 1) {
      EconomyCandidate reinforce;
      reinforce.label = "reinforce";
      reinforce.moves.emplace_back(state.warriors.front().id, map.my_hq);
      reinforce.score = 40.0f + static_cast<float>(summary.own_warriors) * 0.5f;
      candidates.push_back(reinforce);
    }

    return candidates;
  }
};

class AttackPlanner {
public:
  std::vector<AttackCandidate> plan(const AnalyzerOutput &summary,
                                    const GameState &state,
                                    const GameMap &map,
                                    const Paths &) const {
    std::vector<AttackCandidate> candidates;

    AttackCandidate hold;
    hold.label = "hold";
    hold.score = 55.0f;
    if (summary.enemy_hq_hp <= 12) {
      hold.score += 12.0f;
    }
    candidates.push_back(hold);

    if (summary.opening_turn) {
      AttackCandidate opening_push;
      opening_push.label = "opening_push";
      for (const auto &warrior : state.warriors) {
        if (warrior.id.side != map.my_side)
          continue;
        opening_push.moves.emplace_back(warrior.id, map.opp_hq);
      }
      opening_push.score = 85.0f + static_cast<float>(summary.own_warriors) * 2.0f;
      if (summary.enemy_warriors <= 2) {
        opening_push.score += 8.0f;
      }
      candidates.push_back(opening_push);
    }

    if (!summary.opening_turn && summary.own_warriors >= 2) {
      AttackCandidate pressure;
      pressure.label = "pressure";
      int assigned = 0;
      for (const auto &warrior : state.warriors) {
        if (warrior.id.side != map.my_side)
          continue;
        if (assigned >= 2)
          break;
        pressure.moves.emplace_back(warrior.id, map.opp_hq);
        ++assigned;
      }
      pressure.score = 72.0f + static_cast<float>(summary.own_warriors) * 1.0f;
      if (summary.under_pressure) {
        pressure.score += 7.0f;
      }
      candidates.push_back(pressure);
    }

    if (summary.own_warriors >= 3 && summary.enemy_hq_hp > 0) {
      AttackCandidate multi_pressure;
      multi_pressure.label = "multi_pressure";
      int assigned = 0;
      for (const auto &warrior : state.warriors) {
        if (warrior.id.side != map.my_side)
          continue;
        if (assigned >= 3)
          break;
        multi_pressure.moves.emplace_back(warrior.id, map.opp_hq);
        ++assigned;
      }
      multi_pressure.score = 76.0f + static_cast<float>(summary.own_warriors) * 0.8f;
      if (summary.enemy_warriors >= summary.own_warriors) {
        multi_pressure.score -= 4.0f;
      }
      candidates.push_back(multi_pressure);
    }

    return candidates;
  }
};

class DefensePlanner {
public:
  std::vector<DefenseCandidate> plan(const AnalyzerOutput &summary,
                                     const GameState &state,
                                     const GameMap &map,
                                     const Paths &) const {
    std::vector<DefenseCandidate> candidates;

    DefenseCandidate hold;
    hold.label = "hold";
    hold.score = 60.0f;
    if (summary.my_hq_hp <= 10) {
      hold.score -= 8.0f;
    }
    candidates.push_back(hold);

    if (summary.under_pressure) {
      DefenseCandidate stabilize;
      stabilize.label = "stabilize";
      stabilize.upgrades.push_back(map.my_hq);
      for (const auto &warrior : state.warriors) {
        if (warrior.id.side != map.my_side)
          continue;
        stabilize.moves.emplace_back(warrior.id, map.my_hq);
      }
      stabilize.score = 78.0f - static_cast<float>(summary.pressure_score) * 1.5f;
      if (summary.my_hq_hp <= 12) {
        stabilize.score += 10.0f;
      }
      candidates.push_back(stabilize);
    }

    if (summary.enemy_warriors > summary.own_warriors + 1) {
      DefenseCandidate reinforce_hq;
      reinforce_hq.label = "reinforce_hq";
      reinforce_hq.upgrades.push_back(map.my_hq);
      reinforce_hq.score = 72.0f - static_cast<float>(summary.enemy_warriors - summary.own_warriors) * 2.0f;
      candidates.push_back(reinforce_hq);
    }

    return candidates;
  }
};

class Judgement {
public:
  PlanBundle select(const AnalyzerOutput &summary, const GameState &state,
                    const GameMap &map, int,
                    const std::vector<EconomyCandidate> &economies,
                    const std::vector<AttackCandidate> &attacks,
                    const std::vector<DefenseCandidate> &defenses) const {
    PlanBundle best;
    float best_score = -std::numeric_limits<float>::infinity();

    for (const auto &economy : economies) {
      for (const auto &attack : attacks) {
        for (const auto &defense : defenses) {
          float score = score_combination(summary, state, map, economy, attack, defense);
          if (score > best_score) {
            best_score = score;
            best.economy = economy;
            best.attack = attack;
            best.defense = defense;
          }
        }
      }
    }

    return best;
  }

private:
  mutable Model rl_model_;
  mutable bool rl_model_loaded_ = false;

  const Model &get_model() const {
    if (!rl_model_loaded_) {
      rl_model_ = load_rl_model("models/weights.txt");
      rl_model_loaded_ = true;
    }
    return rl_model_;
  }

  std::vector<float> build_features(const AnalyzerOutput &summary,
                                    const EconomyCandidate &economy,
                                    const AttackCandidate &attack,
                                    const DefenseCandidate &defense) const {
    std::vector<float> features;
    features.reserve(16);
    features.push_back(static_cast<float>(summary.turn) / 200.0f);
    features.push_back(std::min(1.0f, static_cast<float>(summary.gold) / 5000.0f));
    features.push_back(std::min(1.0f, static_cast<float>(summary.own_warriors) / 10.0f));
    features.push_back(std::min(1.0f, static_cast<float>(summary.enemy_warriors) / 10.0f));
    features.push_back(economy.train_n > 0 ? 1.0f : 0.0f);
    features.push_back(std::min(1.0f, static_cast<float>(economy.upgrades.size()) / 2.0f));
    features.push_back(std::min(1.0f, economy.score / 100.0f));
    features.push_back(static_cast<float>(attack.moves.size()) / 4.0f);
    features.push_back(attack.label.find("push") != std::string::npos ? 1.0f : 0.0f);
    features.push_back(attack.label.find("pressure") != std::string::npos ? 1.0f : 0.0f);
    features.push_back(std::min(1.0f, static_cast<float>(defense.upgrades.size()) / 2.0f));
    features.push_back(static_cast<float>(defense.moves.size()) / 4.0f);
    features.push_back(defense.label.find("stabilize") != std::string::npos ? 1.0f : 0.0f);
    features.push_back(std::min(1.0f, attack.score / 100.0f));
    features.push_back(std::min(1.0f, defense.score / 100.0f));
    features.push_back(std::min(1.0f, economy.score / 100.0f));
    return features;
  }

  float score_combination(const AnalyzerOutput &summary,
                          const GameState &state,
                          const GameMap &map,
                          const EconomyCandidate &economy,
                          const AttackCandidate &attack,
                          const DefenseCandidate &defense) const {
    float score = economy.score + attack.score + defense.score;
    const auto rl_summary = get_latest_rl_summary();

    if (summary.opening_turn && !attack.moves.empty()) {
      score += 20.0f;
    }
    if (summary.under_pressure) {
      score += 8.0f;
    }
    if (economy.train_n > 0 && summary.turn > 1) {
      score += 6.0f;
    }
    if (!defense.upgrades.empty()) {
      score += 5.0f;
    }
    if (!attack.moves.empty() && summary.turn > 1) {
      score += 3.0f;
    }
    if (summary.my_hq_hp <= 12 && !defense.moves.empty()) {
      score += 7.0f;
    }
    if (summary.enemy_hq_hp <= 12 && !attack.moves.empty()) {
      score += 6.0f;
    }

    const int friendly_buildings = static_cast<int>(std::count_if(state.buildings.begin(), state.buildings.end(), [&](const Building &b) {
      return b.side == map.my_side;
    }));
    if (friendly_buildings < 2 && economy.train_n > 0) {
      score += 3.0f;
    }

    if (rl_summary.valid) {
      score += std::min(8.0f, rl_summary.eval_match_rate * 6.0f);
      if (rl_summary.final_loss < 2.0f) {
        score += 3.0f;
      }
      if (rl_summary.epochs >= 100) {
        score += 1.0f;
      }
    }

    const auto features = build_features(summary, economy, attack, defense);
    const auto logits = predict_rl_logits(get_model(), features);
    if (!logits.empty()) {
      const float rl_score = *std::max_element(logits.begin(), logits.end());
      const float rl_weight = 0.05f + std::min(0.08f, rl_summary.eval_match_rate * 0.04f);
      score += rl_score * rl_weight;
    }

    return score;
  }
};

class Assembler {
public:
  Actions assemble(const AnalyzerOutput &, const PlanBundle &bundle) const {
    Actions out;

    out.train_n = bundle.economy.train_n;
    for (int region : bundle.economy.upgrades) {
      add_unique_upgrade(out, region);
    }
    for (int region : bundle.defense.upgrades) {
      add_unique_upgrade(out, region);
    }

    for (const auto &move : bundle.attack.moves) {
      add_unique_move(out, move);
    }
    for (const auto &move : bundle.defense.moves) {
      add_unique_move(out, move);
    }

    return out;
  }

private:
  static void add_unique_upgrade(Actions &out, int region) {
    if (std::find(out.upgrades.begin(), out.upgrades.end(), region) ==
        out.upgrades.end()) {
      out.upgrades.push_back(region);
    }
  }

  static void add_unique_move(Actions &out,
                              const std::pair<WarriorId, int> &move) {
    for (const auto &existing : out.moves) {
      if (existing.first == move.first) {
        return;
      }
    }
    out.moves.push_back(move);
  }
};

} // namespace strategy

static Actions decide(const GameState &S, const GameMap &M, const Paths &P,
                      int turn) {
  strategy::Analyzer analyzer;
  strategy::EconomyPlanner economy_planner;
  strategy::AttackPlanner attack_planner;
  strategy::DefensePlanner defense_planner;
  strategy::Judgement judgement;
  strategy::Assembler assembler;

  const auto summary = analyzer.analyze(S, M, P, turn);
  const auto economy_candidates = economy_planner.plan(summary, S, M);
  const auto attack_candidates = attack_planner.plan(summary, S, M, P);
  const auto defense_candidates = defense_planner.plan(summary, S, M, P);
  const auto selected = judgement.select(summary, S, M, turn, economy_candidates,
                                         attack_candidates, defense_candidates);
  return assembler.assemble(summary, selected);
}

int main() {
  GameMap M;
  GameState S;
  parse_init(M, S);              // initialize the game
  Paths P = calculate_paths(M); // calculate the shortest paths

  int turn;
  while (read_turn_start(turn)) {
    Actions a = decide(S, M, P, turn);
    emit_command();
    emit_actions(a);
    emit_end();
    read_turn_result(S, M, a);
  }
  return 0;
}
