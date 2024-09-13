// import Board from './board'
import Kinematics from './pages/kinematics';

export default function Home() {
  
  const kinematicsConf = {
    offset: {
      x: 400, y: 300
    },
    arms: [
      {
        L1:100,
        L2:100,
        position: {x:0, y:0},
        color: 'green',
      },
      {
        L1:100,
        L2:100,
        position: {x:100,y:0},
        color: 'red',
        opposite: true
      }
    ],
    init: {
        x:50, y:50
    }
  }

  return (
    <main>
      {/* <Board/> */}
      <Kinematics conf={kinematicsConf}/>
    </main>
  );
}
