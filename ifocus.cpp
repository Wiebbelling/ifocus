#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//#include <omp.h>
#include <unistd.h>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#define DEBUG_NONE 0
#define DEBUG_TEXT 1
#define DEBUG_GTK  2

#ifndef DEBUG
#define DEBUG DEBUG_NONE
#endif

#ifndef HOUGH_MEANS
#define HOUGH_MEANS 1
#endif

#ifndef X_THRESHOLD
#define X_THRESHOLD 3
#endif

#ifndef Y_THRESHOLD
#define Y_THRESHOLD 3
#endif

#ifndef MIN_EYE_SIZE
#define MIN_EYE_SIZE 20
#endif

#ifndef MAX_EYE_SIZE
#define MAX_EYE_SIZE 3 * MIN_EYE_SIZE
#endif

/* Idealmente setar para o FPS da câmera */
#ifndef SLEEP_MS
#define SLEEP_MS 30
#endif

#define ABS(x) ((x) > 0 ? (x) : -(x))

#define MAX_CIRCLES 5

/* Centros das últimas HOUGH_MEANS detecções de íris */
int centers_x[HOUGH_MEANS] = { 0 };
int centers_y[HOUGH_MEANS] = { 0 };
uint8_t cur_center;

cv::Point g_last_pos;
int g_last_leftmost_tl = INT_MAX;
uint8_t g_ignored_lefts = 0;

/*
 * Como o movimento do olho vai e volta, temos que considerar apenas a idea.
 * Portanto, define um número de círculos a ignorar após uma mudança de foco.
 */
uint16_t g_skipped_detects = 0;
#ifndef SKIP_DETECTS
#define SKIP_DETECTS HOUGH_MEANS
#endif

/*
 * Dado um vetor com as coordenadas de círculos já detectados e a imagem do
 * olho já em preto e branco e alto contraste, retorna o círculo que é a íris.
 */
cv::Vec3f
get_iris(cv::Mat &eye, std::vector<cv::Vec3f> &circles)
{
  uint8_t ans = 0;
  /*
   * Se detectamos múltiplos círculos, é preciso detectar qual deles é a íris.
   *
   * Nossa imagem é preto e branca. Ao somar os valores de todos pxs que estão
   * dentro de determinado círulo, estamos somando os valores de branco. Como
   * múltiplos círculos podem ser detectados, e queremos apenas o da íris,
   * vamos selecionar aquele com a menor soma - o mais preto.
   */
  if (circles.size() > 1) {
    /*
     * O máximo é 640 * 480 * 255 = 78336000, cabe num uint32_t (2^32 - 1 =
     * 4294967295). Assume que teremos no máximo MAX_CIRCLES detectados, para
     * evitar alocação dinâmica.
     */
    uint32_t sums[MAX_CIRCLES];
    /*
     * Vide https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html?highlight=accessing%20element
     * Esse é o método mais eficiente de varrer a imagem. Como nossa imagem é
     * 640x480 usamos um uint16. Also, não esperamos muitos círculos, então
     * uint8_t.
     */
    uint8_t ncircles = (uint8_t)(circles.size() > MAX_CIRCLES ? MAX_CIRCLES : circles.size());
    // TODO paralelizar
    for (uint16_t y = 0; y < eye.rows; y++) {
      for (uint16_t x = 0; x < eye.cols; x++) {
        for (uint8_t i = 0; i < ncircles; i++) {
          uint32_t xx = (uint32_t)(circles[i][0]),
                   yy = (uint32_t)(circles[i][1]),
                    r = (uint32_t)(circles[i][2]);
          /* eye.ptr espera a coordenada do início de uma linha, não um px */
          uchar *ptr = eye.ptr<uchar>(y);
          /* Calcula os deltas e vê se o ponto está dentro do círculo */
          uint32_t dx = x - xx,
                   dy = y - yy;
          if ((dx * dx) + (dy * dy) < (r * r))
            sums[i] += uint32_t(ptr[x]);
        }
      }
    }
    uint32_t min = UINT32_MAX;
    for (uint8_t i = 0; i < ncircles; i++) {
      if (sums[i] < min) {
        min = sums[i];
        ans = i;
      }
    }
  }
  return circles[ans];
}

/*
 * Retorna o olho esquerdo.
 */
int
get_leftmost_eye(std::vector<cv::Rect> &eyes)
{
  int leftmost = INT_MAX;
  int ans = 0;
  for (int i = 0; i < (int)(eyes.size()); i++) {
    if (eyes[i].tl().x < leftmost) {
      leftmost = eyes[i].tl().x;
      ans = i;
    }
  }
  /*
   * Se o novo olho esquerdo está muito à direita do anterior, possivelmente
   * é porque não detectamos o anterior nessa passada. Todavia, pode ser que
   * a face tenha se movido. Ignora duas passadas em que a detecção ficou muito
   * à direita, depois para de ignorar.
   */
  if (leftmost > (g_last_leftmost_tl + (g_last_leftmost_tl / 3)) &&
      g_ignored_lefts < 2) {
#if DEBUG >= DEBUG_TEXT
    printf("Olho está muito à direita, ignorando por ora.\n");
#endif
    g_ignored_lefts++;
    return -1;
  } else {
    g_ignored_lefts = 0;
    g_last_leftmost_tl = leftmost;
    return ans;
  }
}

/*
 * Retorna o centro médio, dados os últimos detectados
 */
void
mean_center(cv::Point &center)
{
  uint8_t actually_done = 0;
  for (uint8_t i = 0; i < HOUGH_MEANS; i++)
    if ((centers_x != 0) || (centers_y != 0)) {
      center.x += centers_x[i];
      center.y += centers_y[i];
      actually_done++;
    }
  center.x /= actually_done;
  center.y /= actually_done;
}

/*
 * Muda o foco da janela de acordo com a nova posição; atualiza a antiga.
 */
void
change_focus(cv::Point &pos)
{
  cv::Point delta;
  delta.x = (pos.x - g_last_pos.x);
  delta.y = (pos.y - g_last_pos.y);
  /*
   * Move o foco apenas se o usuário mexeu o suficiente o olho e se já pulamos
   * detecções o suficiente desde a última mudança de foco.
   */

  if (g_skipped_detects >= SKIP_DETECTS) 
  {
    printf("Original: %d %d\n", delta.x, delta.y);
    printf("Power: %f %f\n", pow(delta.x,2), pow(delta.y,2));
    if ((ABS(pow(delta.x,2)) >= X_THRESHOLD) || (ABS(pow(delta.y,2)) >= Y_THRESHOLD)) 
    {
      g_skipped_detects = 0;
      if ((pow(delta.x,2) < 0) || (pow(delta.y,2) < 0)) {
        // system("xdotool key alt+k");
#if DEBUG >= DEBUG_TEXT
        printf("SOBE\n");
#endif
      } else {
        // system("xdotool key alt+j");
#if DEBUG >= DEBUG_TEXT
        printf("DESCE\n");
#endif
      }
    }
  } else {
    g_skipped_detects++;
#if DEBUG >= DEBUG_TEXT
    printf("Acabamos de mover o foco, esperando o olho voltar à posição central\n");
#endif
  }
  /* Atuliza globais */
  g_last_pos = pos;
}

void
detect_and_react(cv::Mat &frame, cv::CascadeClassifier &face_classifier, cv::CascadeClassifier &eye_classifier)
{
  /* Deixa em preto e branco, para detecção */
  cv::Mat frame_pb;
  cv::cvtColor(frame, frame_pb, CV_BGR2GRAY);
  /* Aumenta o contraste pra nos ajudar */
  cv::equalizeHist(frame_pb, frame_pb);
  /*
   * Detecta a(s) face(s)
   * Vide https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
   */
  std::vector<cv::Rect> faces;
  face_classifier.detectMultiScale(frame_pb, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(150, 150));
  /*                                            ^    ^  ^                    ^ Retângulo mínimo da face na imagem
   *                                            |    |  +- Flags para o classificador
   *                                            |    +- Fator para a detecção, quanto maior, menos chance de falso positivo, mas também de positivo
   *                                            +- Fator de escala, usa o padrão
   */
  /* Se nenhuma face foi detectada, segue a vida. Mais frames virão. */
  if (faces.size() < 1)
    return;
  /* Utiliza apenas uma das faces detectadas - não faz usar mais de uma */
  cv::Mat face = frame_pb(faces[0]);
  /*
   * Detecta os olhos.
   * Vide https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
   */
  std::vector<cv::Rect> eyes;
  eye_classifier.detectMultiScale(face, eyes, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(MIN_EYE_SIZE, MIN_EYE_SIZE), cv::Size(MAX_EYE_SIZE, MAX_EYE_SIZE));
#if DEBUG >= DEBUG_GTK
  /* Desenha um retângulo azul na face */
  cv::rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
#endif
  /* Precisamos de pelo menos um olho */
  if (eyes.size() < 1)
    return;
  /* Utiliza apenas um dos olhos */
  int leftmost = get_leftmost_eye(eyes);
  if (leftmost < 0)
    return;
  cv::Rect eye_rect = eyes[leftmost];
#if DEBUG >= DEBUG_GTK
  /* Desenha um retângulo verde no olho sendo usado */
  //for (cv::Rect &eye : eyes)
  //  cv::rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
  cv::rectangle(frame, faces[0].tl() + eye_rect.tl(), faces[0].tl() + eye_rect.br(), cv::Scalar(0, 255, 0), 2);
#endif
  /* Dada a imagem maior (face), cropa o olho */
  cv::Mat eye = face(eye_rect);
  /* Aumenta o contraste */
  cv::equalizeHist(eye, eye);
  //cv::medianBlur(eye, eye, 3);
  //cv::Scalar mean = cv::mean(eye);
  //cv::Mat mask;
  //cv::inRange(eye, mean * 0.8, 255, mask);
  //mask = 255 - mask;
  //cv::imshow("Mask", mask);
  /*
   * Detecta o(s) círculo(s)
   * Vide https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
   *
   * Os valores para mínimo e máximo foram retirados de
   * https://picoledelimao.github.io/blog/2017/01/28/eyeball-tracking-for-mouse-control-in-opencv/
   *
   * ... que utiliza valores otimizados para a resolução com que trabalhamos
   */
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(eye, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
  /*                             ^                  ^  ^             ^    ^   ^             ^ Raio máximo
   *                             |                  |  |             |    |   +- Raio mínimo do círculo
   *                             |                  |  |             |    +- Área mínima do círculo
   *                             |                  |  |             +- Threshold de detecção
   *                             |                  |  +- Distância mínima entre os círculos
   *                             |                  +- Valor padrão
   *                             +- Método a ser usado para a detecção
   */
  if (circles.size() > 0) {
    /* Pega a íris a partir de possivelmente múltiplos círculos (em geral 1) */
    cv::Vec3f iris = get_iris(eye, circles);
    /*
     * O método para detectar círculos é bem instável, portanto determina o
     * centro do círculo como uma média dos centros dos últimos círculos.
     */
    centers_x[cur_center] = (int)(iris[0]);
    centers_y[cur_center] = (int)(iris[1]);
    cur_center = (uint8_t)((cur_center + 1) % HOUGH_MEANS);
    cv::Point center;
    mean_center(center);
    change_focus(center);
#if DEBUG >= DEBUG_GTK
    /* Desenha o círculo na imagem da face */
    cv::circle(frame, faces[0].tl() + eye_rect.tl() + center, (int)(iris[2]), cv::Scalar(0, 0, 255), 2);
    /* Desenha o círculo na imagem do olho */
    cv::circle(eye, center, (int)(iris[2]), cv::Scalar(255, 255, 255), 2);
#endif
  }
#if DEBUG >= DEBUG_GTK
  /* Exibe a imagem do olho */
  cv::imshow("Eye", eye);
  /* Move a janela para não ficar embaixo da maior */
  cv::moveWindow("Eye", 900, 300);
#endif
}

void
capture_frame(cv::VideoCapture &cap, cv::Mat &frame)
{
#ifdef INVERT_AXIS
  cv::Mat raw;
  cap >> raw;
  cv::flip(raw, frame, 1);
#else
  cap >> frame;
#endif
}

int
main(void)
{
  cur_center = 0;
  cv::CascadeClassifier face_classifier;
  cv::CascadeClassifier eye_classifier;
  if (!face_classifier.load("./haarcascade_frontalface_alt.xml") || !eye_classifier.load("./haarcascade_eye_tree_eyeglasses.xml")) {
    fprintf(stderr, "Missing files.\n");
    exit(EXIT_FAILURE);
  }

  cv::VideoCapture cap;
  cap.open(0);

//  cv::VideoCapture cap(-1);
  
  if (!cap.isOpened()) {
    fprintf(stderr, "No webcam found.\n");
    exit(EXIT_FAILURE);
  }

  cv::Mat frame;
  capture_frame(cap, frame);
  bool done = !frame.data;
  while (!done) {
    detect_and_react(frame, face_classifier, eye_classifier);
#if DEBUG >= DEBUG_GTK
    cv::imshow("Webcam", frame);
    int keypress = cv::waitKey(SLEEP_MS);
#else
    usleep(SLEEP_MS);
#endif
    capture_frame(cap, frame);
#if DEBUG >= DEBUG_GTK
    done = (!(frame.data)) || (keypress == 81) || (keypress == 113);
    /*                                     ^ Q                 ^ q */
#else
    done = !(frame.data);
#endif
  }
  return 0;
}


