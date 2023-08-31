#include <Eigen/Dense>

#include <optional>
#include <vector>

using Pos = Eigen::Vector2d;
using PosInput = Eigen::Ref<const Pos>;
using ActionList = Eigen::MatrixXd;
using ActionListInput = Eigen::Ref<const ActionList>;
using PosList = Eigen::MatrixXd;

class Mouse {
  public:
    Mouse(const PosInput& init_pos, double init_ori, const ActionListInput& possible_actions, bool save_history = true);

    // properties
    Pos get_position() const;
    Pos get_initial_position() const;
    double get_orientation() const;
    double get_initial_orientation() const;
    std::vector<Pos> get_history_position() const;
    std::vector<double> get_history_orientation() const;

    void enable_history(bool enable = true);

    void reset(std::optional<PosInput> initial_pos = std::nullopt, std::optional<double> initial_ori = std::nullopt);

    void reset_history();

    PosList get_endpoints() const;

    bool move_to(const PosInput& new_pos);

    bool move_to(double x, double y);

  private:
    Pos init_pos;
    Pos pos;
    double init_ori;
    double ori;
    ActionList possible_actions;
    bool save_history;
    std::vector<Pos> history_pos;
    std::vector<double> history_ori;
};