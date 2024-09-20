export interface Kinematics<T> {
  getConf: () => T;
  setConf: (conf: T) => void;
  getPoints: (point: Point) => Point[];
  getAngle: (point: Point) => number;
};

export interface Arm<T> {
  kinematics: Kinematics<T>;
  getConf: () => T;
  setConf: (conf: T) => void;
  getType: () => string;
};

export interface ArmParams {
  L1: number;
  type: string;
};

export abstract class ArmConfigurable<T extends ArmParams> implements Arm<T> {
  kinematics: Kinematics<T>;
  constructor(kinematics: Kinematics<T>) {
    this.kinematics = kinematics;
  }
  getConf() {
    return this.kinematics.getConf();
  };
  setConf(conf: T) {
    this.kinematics.setConf(conf);
  };
  abstract getType(): string;
};

export abstract class NoneKinematics<T extends ArmParams> implements Kinematics<T> {
  conf: T;
  constructor(conf: T) {
    this.conf = conf;
  }
  getConf() {
    return this.conf;
  };
  setConf(conf: T) {
    this.conf = conf;
  }
  abstract getPoints(point: Point): Point[];
  abstract getAngle(point: Point): number;
}

export interface Point {
  x: number;
  y: number;
};

export const x0y0: Point = { x: 0, y: 0 };