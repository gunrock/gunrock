#include<cstdio>
#include<vector>
#include<algorithm>
#include<cstdlib>
#include<set>
#include<time.h>

using namespace std;

struct Point{
	int x;
	int y;

	Point(){}
	Point(int X, int Y):x(X), y(Y){}

	Point operator+(const Point &p){
		return Point(x+p.x, y+p.y);
	}
	bool operator < (const Point &p1) const{
		return (p1.y > y) or (p1.y == y and p1.x < x);
	}
};

struct comp{
	inline bool operator()(const Point &p1, const Point &p2){
		return (p1.y > p2.y) or (p1.y == p2.y and p1.x < p2.x);
	}
};

struct star{
	Point o;
	Point left;
	Point right;
	Point up;
	Point down;
	
	star(Point O, Point Left, Point Right, Point Up, Point Down){
		o = O;
        left = Left;
        right = Right;
        up = Up;
        down = Down;
    }
    star(star S, int x, int y){
        o 	    = S.o		+ Point(x, y);
        left 	= S.left 	+ Point(x, y);
        right 	= S.right 	+ Point(x, y);
        up 	    = S.up 		+ Point(x, y);
        down 	= S.down 	+ Point(x, y);
    }
};
	
void generate(set<Point>& stars, star S, int dim1, int dim2, int n){
	srand(time(NULL));
	for (int i=0; i<n; ++i){
		int x = 1+rand()%(dim1-2);
		int y = 1+rand()%(dim2-2);
		star ns = star(S, x, y); 
		stars.insert(ns.o);
		stars.insert(ns.up);
		stars.insert(ns.down);
		stars.insert(ns.left);
		stars.insert(ns.right);
	}
}
void draw(vector<Point>& points, int dim1, int dim2, int n){
	sort(points.begin(), points.end(), comp());
	for (int i=0; i<points.size(); ++i)
		fprintf(stderr, "(%d,%d) ", points[i].x, points[i].y);
	fprintf(stderr, "\n");
	int iterator = 0;
	fprintf(stderr, "|_|");
	for (int i=1; i<=dim1; ++i)
		fprintf(stderr, "%2.d ", i);
    fprintf(stderr, "\n");
	int O = 1;
	int iterator_last = 0;
	for (int i=dim2; i>0; i--){
		fprintf(stderr, "%2.d|", i);
		for (int j=1; j<=dim1; ++j){
			if (points[iterator].y == i and points[iterator].x == j){
				fprintf(stderr, "%2.d ", O);
				++iterator;
			}else{
				fprintf(stderr, "   ");
			}
		}
		fprintf(stderr, "\n");
	}
}
void write_out(vector<Point>& points){
	sort(points.begin(), points.end(), comp());
	printf("%d 2\n", points.size());
	for (int i=0; i<points.size(); ++i)
		printf("%d %d %d\n", i+1, points[i].x, points[i].y);
}
int main(){
	int dim1, dim2, n;
	Point o(1, 1);
	Point left(0, 1);
	Point right(2, 1);
	Point up(1, 2);
	Point down(1, 0);
	star S(o, left, right, up, down);
	scanf("%d %d %d", &dim1, &dim2, &n);
	set<Point> points_set;
	generate(points_set, S, dim1, dim2, n);
	vector<Point> points(points_set.size());
	copy(points_set.begin(), points_set.end(), points.begin());
	draw(points, dim1, dim2, n);
	write_out(points);
	return 0;
}
