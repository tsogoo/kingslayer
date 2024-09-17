import { createContext } from "react";

export interface Arm {
  L1: number;
  L2: number;
  position: Point;
  color: string;
  opposite?: boolean;
};

export interface Point {
  x: number;
  y: number;
};

export interface Animation {
  start: Point;
  end: Point;
  duration: number;
}

export interface Animate {
  point: Point;
  delay: number;
}

export interface Config {
  type: string;
  val: number|string;
}

interface KinematicsContextState {
  position: Point;
  offset: Point;
  setPosition: (point: Point) => void;
};

export const x0y0:Point = {x:0,y:0};

export const KinematicsContext = createContext<KinematicsContextState>({
  position: x0y0,
  offset: x0y0,
  setPosition: (point: Point) => {},
});

interface ValueContextState {
  setValue: (type: string, val: string|number) => void;
};

interface AnimationContextState extends ValueContextState {
}

export const AnimationContext = createContext<AnimationContextState>({
  setValue: (type: string, val: string|number) => {}
})