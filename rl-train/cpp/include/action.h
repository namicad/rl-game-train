#pragma once

enum class ActionType
{
    MOVE,
    ATTACK,
    BUILD,
    TRAIN,
    UPGRADE,
    END
};

struct Action
{
    ActionType type = ActionType::END;
    int unit_id = -1;
    int target = -1;
};