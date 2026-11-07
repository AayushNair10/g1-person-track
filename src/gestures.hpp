// 15 static hand gestures from the 21 hand landmarks, by geometric rules (no extra model):
//   Open palm, Fist, Thumbs up, Thumbs down, Pointing up / down / left / right, Victory, Three, OK, Pinch,
//   Rock, I love you, Call me.
// Finger states come from the 3-D world landmarks (metres, so independent of distance and image rotation):
// a finger is straight when its two upper joints are nearly unbent and its tip is farther from the wrist
// than its middle joint, curled when strongly bent or its tip comes back near the knuckles. Directions
// (thumbs up/down, pointing) come from the image landmarks. "left"/"right" are as seen by the camera.
// Landmarks: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; little 17-20 (base -> tip).
#pragma once

#include <deque>
#include <string>

#include "hands.hpp"

namespace pt {

enum class Finger : int { Curled = 0, Half = 1, Straight = 2 };

struct GestureConfig {
    float straight_bend_deg = 50.f;  // sum of the two upper-joint angles below this: straight
    float curled_bend_deg = 100.f;   // above this: curled
    float thumb_straight_deg = 45.f; // thumb: sum of its two joint angles
    float thumb_out = 0.75f;         // thumb tip from the index base, in palm lengths: farther = thumb out
    float touch = 0.30f;             // thumb tip to index tip, in palm lengths: closer = touching
    float dir_ratio = 0.6f;          // a direction counts when its axis carries this share of the vector
};

struct HandShape {
    Finger f[5];         // thumb, index, middle, ring, little
    float palm = 0.f;    // wrist -> middle-finger base (metres)
    float touch = 0.f;   // thumb tip -> index tip, in palm lengths
};

HandShape hand_shape(const HandLandmarks& h, const GestureConfig& c = GestureConfig());
// "" when the hand shows none of the 15 gestures
std::string classify_gesture(const HandLandmarks& h, const GestureConfig& c = GestureConfig());

// Reports a gesture once it is the label of at least `need` of the last `window` frames ("" otherwise).
class GestureSmoother {
public:
    explicit GestureSmoother(int window = 7, int need = 4) : window_(window), need_(need) {}
    const std::string& update(const std::string& label);
    const std::string& stable() const { return stable_; }

private:
    int window_, need_;
    std::deque<std::string> hist_;
    std::string stable_;
};

}  // namespace pt
