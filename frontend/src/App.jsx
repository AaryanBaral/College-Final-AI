import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Profile from "./pages/Profile.jsx";
import Signin from "./pages/Signin.jsx";
import Signup from "./pages/Signup.jsx";
import Create from "./pages/Create.jsx";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

// export default function App() {
//   return (
//     <BrowserRouter>
//         <Routes>
//             <Route path='/' element={<Home/>}></Route>
//             <Route path='/profile' element={<Profile/>}></Route>
//             <Route path='/signin' element={<Signin/>}></Route>
//             <Route path='/signup' element={<Signup/>}></Route>
//             <Route path='/create' element={<Create/>}></Route>
//             {/* <Route path='/predict' element={<Predict/>}></Route> */}
//             {/* <Route path='/profile/settings' element={<Settings/>}></Route> */}
//         </Routes>
//             <ToastContainer position="top-center" autoClose={3000} />
//     </BrowserRouter>
//   );
// }
export default function App() {
  return (
    <BrowserRouter>
      <AnimatedRoutes />
      <ToastContainer position="top-center" autoClose={3000} />
    </BrowserRouter>
  );
}

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={
            <PageTransition>
              <Home />
            </PageTransition>
          }
        />
        <Route
          path="/profile"
          element={
            <PageTransition>
              <Profile />
            </PageTransition>
          }
        />
        <Route
          path="/signin"
          element={
            <PageTransition>
              <Signin />
            </PageTransition>
          }
        />
        <Route
          path="/signup"
          element={
            <PageTransition>
              <Signup />
            </PageTransition>
          }
        />
        <Route
          path="/create"
          element={
            <PageTransition>
              <Create />
            </PageTransition>
          }
        />
      </Routes>
    </AnimatePresence>
  );
}

// 🔥 Reusable animation wrapper
function PageTransition({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, backgroundColor: "#000" }}
      animate={{ opacity: 1, backgroundColor: "#000" }}
      exit={{ opacity: 0, backgroundColor: "#000" }}
      transition={{
        duration: 0.4,
        ease: "easeIn",
      }}
      style={{
        position: "absolute",
        width: "100%",
        height: "100%",
        backgroundColor: "#000",
        color: "#000000ff",
        top: 0,
        left: 0,
      }}
    >
      {children}
    </motion.div>
  );
}
