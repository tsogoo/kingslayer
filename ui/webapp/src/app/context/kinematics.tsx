import { createContext } from "react";
import { Point, x0y0 } from "../common/kinematics"
import { Animation } from "../common/ui";

interface KinematicsContextState {
  position: Point;
  offset: Point;
  setPosition: (point: Point) => void;
  animation: Animation;
  setAnimation: (animation: Animation) => void;
};

export const KinematicsContext = createContext<KinematicsContextState>({
  position: x0y0,
  offset: x0y0,
  setPosition: (point: Point) => { },
  animation: { start: x0y0, end: x0y0, duration: 0 },
  setAnimation: (animation: Animation) => { }
});