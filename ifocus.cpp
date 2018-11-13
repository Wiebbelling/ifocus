#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//#include <omp.h>
#include <unistd.h>

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
#define HOUGH_MEANS 10
#endif

#ifndef X_THRESHOLD
#define X_THRESHOLD 1
#endif

#ifndef Y_THRESHOLD
#define Y_THRESHOLD 100
#endif

#ifndef MIN_EYE_SIZE
#define MIN_EYE_SIZE 30
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

cv::Point g_last_pos;
int g_last_leftmost_tl = INT_MAX;
uint8_t g_ignored_lefts = 0;
#define IGNORE_LEFTS 4

cv::Point
get_iris(cv::Rect &eye, cv::Mat &face)
{
  cv::Point ans;
  ans.x = eye.height / 2;
  ans.y = eye.width  / 2;
  cv::Mat ROI(face, eye);
  cv::Size s;
  cv::Point offset;
  ROI.locateROI(s, offset);
  ans.x += offset.x;
  ans.y += offset.y;
  return ans;
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
      g_ignored_lefts < 4) {
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

#define SCALE_(x) (pow(MAX(1, (x)), 2))

#define SCALE(x) ((x) > 0 ? SCALE_((x)) : -SCALE_(-(x)))


/*
 * Muda o foco da janela de acordo com a nova posição; atualiza a antiga.
 */
void
change_focus(cv::Point &pos)
{
  cv::Point delta;
  delta.x = SCALE(pos.x - g_last_pos.x);
  delta.y = SCALE(pos.y - g_last_pos.y);
  /* Move o foco */
#if DEBUG >= DEBUG_TEXT
  printf("%d %d\n", delta.x, delta.y);
#endif
  char buffer[256];
  snprintf(buffer, 256, "xdotool mousemove_relative -- %d %d", delta.x, delta.y);
  system(buffer);
  /* Atuliza globais */
  g_last_pos = pos;
}

int
detect_and_react(cv::Mat &frame, cv::CascadeClassifier &face_classifier, cv::CascadeClassifier &eye_classifier)
{
  int ans = 1;
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
    return ans;
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
    return ans;
  /* Utiliza apenas um dos olhos */
  int leftmost = get_leftmost_eye(eyes);
  if (leftmost < 0)
    return ans;
  cv::Rect eye_rect = eyes[leftmost];
#if DEBUG >= DEBUG_GTK
  /* Desenha um retângulo verde no olho sendo usado */
  cv::rectangle(frame, faces[0].tl() + eye_rect.tl(), faces[0].tl() + eye_rect.br(), cv::Scalar(0, 255, 0), 2);
#endif
  cv::Point iris = get_iris(eye_rect, face);
  change_focus(iris);
  ans = 0;
#if DEBUG >= DEBUG_GTK
  /* Desenha o círculo na imagem da face */
  cv::circle(frame, iris, MAX(1, eye_rect.height / 10), cv::Scalar(0, 0, 255), -1);
#endif
  return ans;
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
  cv::CascadeClassifier face_classifier;
  cv::CascadeClassifier eye_classifier;
  if (!face_classifier.load("./haarcascade_frontalface_alt.xml") || !eye_classifier.load("./haarcascade_eye_tree_eyeglasses.xml")) {
    fprintf(stderr, "Missing files.\n");
    exit(EXIT_FAILURE);
  }
  cv::VideoCapture cap(-1);
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
#endif
    int keypress = cv::waitKey(SLEEP_MS);
    capture_frame(cap, frame);
    done = (!(frame.data)) || (keypress == 81) || (keypress == 113);
    /*                                     ^ Q                 ^ q */
  }
  return 0;
}


