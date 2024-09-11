// import Board from './board'
import Kinematics from './pages/kinematics';

export default function Home() {
  
  const kinematicsConf = {
    offset: {
      x: 400, y: 300
    },
    arms: [
      {
        L1:200,
        L2:200,
        position: {x:0, y:0},
        color: 'green',
      },
      {
        L1:200,
        L2:200,
        position: {x:100,y:0},
        color: 'red',
        opposite: true
      }
    ]
  }

  return (
    <main>
      {/* <Board/> */}
      <Kinematics conf={kinematicsConf}/>
    </main>
  );
}
