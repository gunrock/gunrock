/**
 * @file points.hxx
 * @author Agnieszka Lupinska (lupinska.agnieszka@gmail.com)
 * @brief Class for generating points for nearest neighbor.
 * @version 0.1
 * @date 2019-11-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <time.h>

using namespace std;

namespace gunrock {
namespace io {

template <typename type_t = int>
struct point_t {
  type_t x;
  type_t y;

  point_t() {}
  point_t(type_t X, type_t Y) : x(X), y(Y) {}

  point_t operator+(const point_t& p) { return point_t(x + p.x, y + p.y); }
  bool operator<(const point_t& p1) const {
    return ((p1.y > y) || ((p1.y == y) && (p1.x < x)));
  }
};

struct comp {
  template <typename type_t = int>
  inline bool operator()(const point_t<type_t>& p1, const point_t<type_t>& p2) {
    return ((p1.y > p2.y) || ((p1.y == p2.y) && (p1.x < p2.x)));
  }
};

template <typename type_t = int>
struct star_t {
  point_t<type_t> o;
  point_t<type_t> left;
  point_t<type_t> right;
  point_t<type_t> up;
  point_t<type_t> down;

  star_t(point_t<type_t> O,
         point_t<type_t> Left,
         point_t<type_t> Right,
         point_t<type_t> Up,
         point_t<type_t> Down) {
    o = O;
    left = Left;
    right = Right;
    up = Up;
    down = Down;
  }

  star_t(star_t<type_t> S, int x, int y) {
    o = S.o + point_t<type_t>(x, y);
    left = S.left + point_t<type_t>(x, y);
    right = S.right + point_t<type_t>(x, y);
    up = S.up + point_t<type_t>(x, y);
    down = S.down + point_t<type_t>(x, y);
  }
};

template <typename type_t = int>
void generate(std::set<point_t<int>>& stars,
              star_t<type_t> S,
              int dim1,
              int dim2,
              int n) {
  srand(time(NULL));
  for (int i = 0; i < n; ++i) {
    int x = 1 + rand() % (dim1 - 2);
    int y = 1 + rand() % (dim2 - 2);
    star_t<type_t> ns = star_t<type_t>(S, x, y);
    stars.insert(ns.o);
    stars.insert(ns.up);
    stars.insert(ns.down);
    stars.insert(ns.left);
    stars.insert(ns.right);
  }
}

template <typename type_t = int>
void draw(std::vector<point_t<type_t>>& points, int dim1, int dim2, int n) {
  sort(points.begin(), points.end(), comp());
  for (int i = 0; i < points.size(); ++i)
    fprintf(stderr, "(%d,%d) ", points[i].x, points[i].y);
  fprintf(stderr, "\n");
  int iterator = 0;
  fprintf(stderr, "|_|");
  for (int i = 1; i <= dim1; ++i)
    fprintf(stderr, "%2.d ", i);
  fprintf(stderr, "\n");
  int O = 1;
  for (int i = dim2; i > 0; i--) {
    fprintf(stderr, "%2.d|", i);
    for (int j = 1; j <= dim1; ++j) {
      if ((points[iterator].y == i) && (points[iterator].x == j)) {
        fprintf(stderr, "%2.d ", O);
        ++iterator;
      } else {
        fprintf(stderr, "   ");
      }
    }
    fprintf(stderr, "\n");
  }
}

template <typename type_t = int>
void write(std::vector<point_t<type_t>>& points) {
  sort(points.begin(), points.end(), comp());
  printf("%d 2\n", (int)points.size());
  for (int i = 0; i < points.size(); ++i)
    printf("%d %d %d\n", i + 1, points[i].x, points[i].y);
}

}  // namespace io
}  // namespace gunrock