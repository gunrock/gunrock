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

#include <gunrock/io/points.hxx>

using namespace gunrock;
using namespace io;

int main() {
  int dim1, dim2, n;
  point_t<int> o(1, 1);
  point_t<int> left(0, 1);
  point_t<int> right(2, 1);
  point_t<int> up(1, 2);
  point_t<int> down(1, 0);
  star_t<int> S(o, left, right, up, down);
  int inputs = std::scanf("%d %d %d", &dim1, &dim2, &n);
  std::set<point_t<int>> points_set;
  generate(points_set, S, dim1, dim2, n);
  std::vector<point_t<int>> points(points_set.size());
  copy(points_set.begin(), points_set.end(), points.begin());
  draw(points, dim1, dim2, n);
  write(points);
}