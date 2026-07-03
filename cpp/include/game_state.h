#pragma once
#include <vector>

struct GameState
{
    int turn = 0;

    int gold = 0;
    int population = 0;

    int city_count = 0;
    int enemy_city_count = 0;

    int kills = 0;
    int losses = 0;

    bool win = false;
    bool lose = false;

    void init()
    {
        turn = 0;
        gold = 100;
        population = 0;

        city_count = 1;
        enemy_city_count = 1;

        kills = 0;
        losses = 0;

        win = false;
        lose = false;
    }

    bool is_terminal() const
    {
        return win || lose || turn > 500;
    }
};