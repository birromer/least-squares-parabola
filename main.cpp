#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>
#include <iostream>

//Valor de Thresholding para definir se o pixel eh branco ou preto
#define LIMIAR_BINARIZACAO 140

//Distancia maxima que um pixel pode estar das linhas detectadas por Hough para ser apagado
#define LIMIAR_HOUGH 50

//Um ponto eh outiler se a media da distancia "X" dele para os outros pontos for maior que LIMIAR_OUTLIER * Largura_da_imagem
#define LIMIAR_OUTLIER 0.3

//Numero de pontos para usar no "Least Square Fitting" da parabola
#define N_POINTS_TO_FIT 3

//Iteracoes do RANSAC
#define RANSAC_MAX_ITERATIONS 1000

//Pi
#define PI 3.1415926536

using namespace cv;

//Remove outliers do conjunto de pontos
std::vector<Point2f> remove_outliers(std::vector<Point2f> points, int distLimit){
  int dist;
  for(int i = 0; i < points.size(); i++){
    dist = 0;
    for(int j = 0; j < points.size(); j++){
      dist = dist + abs(points[i].x - points[j].x);
    }
    dist = dist / points.size();
    if(dist > distLimit)
      points.erase(points.begin() + i);
  }
  return points;
}

//Calcula o erro total da parabola
int totalError(std::vector<double> parabola, std::vector<Point2f> sampledPoints){
  int resultado = 0;

  for(Point2f point : sampledPoints){
    resultado = resultado + abs((parabola[0] + parabola[1]*point.x + parabola[2]*point.x*point.x) - point.y);
  }

  return resultado;
}

//Encontra a coordenada Y do primeiro pixel branco
int getParabolaY(Mat image, int x){
  int y = 0;
  while(y < image.rows && image.at<Vec3b>(y, x)[0] == 0)
    y++;

  if(y == image.rows)
    return -1;
  else
    return y;
}

//Retorna as 2 linhas "mais ortogonais" do conjunto de linhas resultante da transformada Hough
std::vector<Vec2f> getBestLines(std::vector<Vec2f> lines){
  Vec2f bLineA, bLineB;
  Vec2f cLineA, cLineB;
  double bestThetaDif = 10000;

  for(Vec2f line1 : lines){
    for(Vec2f line2 : lines){
      double thetaDif = abs(PI/2 - abs(line1[1] - line2[1]));

      if(thetaDif < bestThetaDif){
        bestThetaDif = thetaDif;
        bLineA = line1;
        bLineB = line2;
      }
    }
  }

  lines.clear();
  lines.push_back(bLineA);
  lines.push_back(bLineB);

  return lines;
}

//Least Square Fitting
std::vector<double> fitParabola(std::vector<Point2f> points){
  double a, b, c;

  Mat A = Mat::ones(points.size(), 3, CV_64F);
  Mat B = Mat::ones(points.size(), 1, CV_64F);

  for(int i = 0; i < points.size(); i++){
    A.at<double>(i, 0) = 1;
    A.at<double>(i, 1) = points[i].x;
    A.at<double>(i, 2) = points[i].x * points[i].x;

    B.at<double>(i, 0) = points[i].y;
  }

  Mat AtA = A.t() * A;
  Mat AtB = A.t() * B;

  Mat X = AtA.inv() * AtB;

  std::vector<double> resultado;

  resultado.push_back(X.at<double>(-1, 1));
  resultado.push_back(X.at<double>(0, 1));
  resultado.push_back(X.at<double>(1, 1));

  return resultado;
}

int main(int argc, char* argv[]){
  Mat image, imageOriginal;

  //Le a imagem
  image = imread(argv[1], IMREAD_GRAYSCALE);
  imageOriginal = image.clone();
  cvtColor(imageOriginal, imageOriginal, COLOR_GRAY2BGR);

  //Thresholding
  for(int i = 0; i < image.rows; i++){
    for(int j = 0; j < image.cols; j++){
      Vec3b pixel = image.at<Vec3b>(i, j);

      if(pixel[0] < LIMIAR_BINARIZACAO)
        image.at<Vec3b>(i, j) = {0, 0, 0};
      else
        image.at<Vec3b>(i, j) = {255, 255, 255};
    }
  }

  //Operacao morfologica
  int morph_elem = 0;
  int morph_size = 1;

  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  morphologyEx( image, image, 1, element );

  //Canny Edge Detection
  Mat cannyEdge;
  Canny(image, cannyEdge, 50, 200, 3);

  image = 255 - image;
  cvtColor(image, image, COLOR_GRAY2BGR);

  //Hough Transform
  std::vector<Vec2f> lines;
  HoughLines(cannyEdge, lines, 1, CV_PI/180, 150, 0, 0 );
  lines = getBestLines(lines);

  //Apaga os eixos da imagem
  for(int i = 0; i < 2; i++)
  {
      float ro = lines[i][0];
      float theta = lines[i][1];
      float thetaDraw = lines[i][1];
      double a, b, x0, y0;

      if(i == 1){
        thetaDraw = lines[0][1] + PI/2;
        if(thetaDraw > PI)
          thetaDraw = thetaDraw - PI;
      }

      Point p1, p2;
      a = cos(theta);
      b = sin(theta);
      x0 = a*ro;
      y0 = b*ro;
      p1.x = cvRound(x0 + 1000*(-b));
      p1.y = cvRound(y0 + 1000*(a));
      p2.x = cvRound(x0 - 1000*(-b));
      p2.y = cvRound(y0 - 1000*(a));
      line( image, p1, p2, Scalar(0,0,0), LIMIAR_HOUGH, LINE_AA);

      a = cos(thetaDraw);
      b = sin(thetaDraw);
      x0 = a*ro, y0 = b*ro;
      p1.x = cvRound(x0 + 1000*(-b));
      p1.y = cvRound(y0 + 1000*(a));
      p2.x = cvRound(x0 - 1000*(-b));
      p2.y = cvRound(y0 - 1000*(a));
      line( imageOriginal, p1, p2, Scalar(255,0,0), 2, LINE_AA);
  }

  //Rotaciona a imagem
  Mat imageRotate;
  double rotAngle = abs(PI/2 - lines[0][1]) < abs(PI/2 - lines[1][1]) ? lines[1][1] : lines[0][1];
  rotAngle = (180*rotAngle)/PI;
  while(rotAngle > 90)
    rotAngle = rotAngle - 180;

  Point2f pc(image.cols/2., image.rows/2.);

  warpAffine(image, imageRotate, getRotationMatrix2D(pc, rotAngle, 1.0), image.size());

  //Fittin por RANSAC
  int pX, pY;

  int totalPoints = imageRotate.cols / 4;
  bool chosenPoints[imageRotate.cols] = {false};

  std::vector<Point2f> sampledPoints;

  for(int i = 0; i < totalPoints; i++){
    pY = -1;
    while(pY == -1 || chosenPoints[pX] == true){
      pX = rand()%image.cols;
      pY = getParabolaY(imageRotate, pX);
    }
    chosenPoints[pX] = true;
    Point2f point(pX, pY);
    sampledPoints.push_back(point);
  }

  sampledPoints = remove_outliers(sampledPoints, imageRotate.cols * LIMIAR_OUTLIER);

  std::vector<double> bestParabola;
  int bestErro = INT_MAX;
  for(int iteration = 0; iteration < RANSAC_MAX_ITERATIONS; iteration++){
    std::vector<Point2f> pointsToFit;
    std::vector<double> parabola;

    for(int i = 0; i < N_POINTS_TO_FIT; i++){
      int fitPoint = rand()%sampledPoints.size();
      pointsToFit.push_back(sampledPoints[fitPoint]);
      circle(imageRotate, sampledPoints[fitPoint], 4 , Scalar(0,255,0), 10);
    }

    parabola = fitParabola(pointsToFit);

    int erroParabola = totalError(parabola, sampledPoints);
    if(erroParabola < bestErro){
      bestParabola = parabola;
      bestErro = erroParabola;
    }
  }

  imageRotate = 0;
  for(int i = 0; i < imageRotate.cols; i++){
    pX = i;
    pY = bestParabola[0] + bestParabola[1]*pX + bestParabola[2]*pX*pX;

    if(pY > 0 && pY < image.rows){
      imageRotate.at<Vec3b>(pY, pX) = {0, 0, 255};
      imageRotate.at<Vec3b>(pY-1, pX) = {0, 0, 255};
      imageRotate.at<Vec3b>(pY+1, pX) = {0, 0, 255};
      imageRotate.at<Vec3b>(pY-2, pX) = {0, 0, 255};
      imageRotate.at<Vec3b>(pY+2, pX) = {0, 0, 255};
    }
  }

  //Inverte a rotacao
  warpAffine(imageRotate, imageRotate, getRotationMatrix2D(pc, -rotAngle, 1.0), imageRotate.size());

  //Desenha a parabola na imagem original
  image = imageOriginal;
  for(int i = 0; i < min(imageOriginal.rows, imageRotate.rows); i++){
    for(int j = 0; j < min(imageOriginal.cols, imageRotate.cols); j++){
      if(imageRotate.at<Vec3b>(i,j)[2] > 128)
        image.at<Vec3b>(i,j) = {0,255,0};
    }
  }

  namedWindow( "Resultado", WINDOW_AUTOSIZE );
  imshow( "Resultado", image );

  imwrite( "Resultado.jpg", image );

  waitKey(0);

  return 0;
}
