import React from "react";
import { FaEye } from "react-icons/fa6";
import { FaGithub } from "react-icons/fa";
import "../styles/navbar.css";
import { Link } from "react-router";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router";
import { toast } from "react-toastify";
import {
  deleteUserFailure,
  deleteUserStart,
  deleteUserSuccess,
} from "../redux/user/userSlice";

export default function Navbar() {
  const reduxData = useSelector((state) => state.user);
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [isLogoutDisabled, setIsLogoutDisabled] = React.useState(false);

  const handleLogout = async (e) => {
    e.preventDefault();
    dispatch(deleteUserStart());
    setIsLogoutDisabled(true);
    try {
      const response = await fetch("/api/auth/signout", {
        method: "DELETE",
        headers: {
          "Content-type": "application/json",
        },
        body: JSON.stringify(reduxData.currentUser),
        credentials: "include",
      });
      const result = await response.json();
      if (result.success == false) {
        dispatch(deleteUserFailure(result.message));
        setIsLogoutDisabled(false);
        toast.error(result.message);
        return;
      }
      dispatch(deleteUserSuccess());
      setIsLogoutDisabled(false);
      toast.success("Logged out successfully.");
      navigate("/");
      return;
    } catch (error) {
      dispatch(deleteUserFailure(error.message));
      setIsLogoutDisabled(false);
      toast.error(error.message);
      return;
    }
  };

  return (
    <nav className="navbar-main">
      <div className="navbar">
        <div className="navbar-left">
          <ul>
            <li>
              <Link className="route-link" to={"/"}>
                <FaEye size={20} />
              </Link>
            </li>
            {reduxData.currentUser && (
              <li>
                <Link to={"/create"} className="route-link">
                  Create
                </Link>
              </li>
            )}
          </ul>
        </div>
        <div className="navbar-center">
          <ul>
            <li>
              <Link
                to={"https://github.com/fuseai-fellowship/Nayan"}
                className="route-link"
              >
                <FaGithub size={24} />
              </Link>
            </li>
          </ul>
        </div>
        <div className="navbar-right">
          <ul>
            {!reduxData.currentUser ? (
              <>
                <li>
                  <Link to={"/signin"} className="route-link">
                    login
                  </Link>
                </li>
                <li>
                  <Link to={"/signup"} className="route-link">
                    sign up
                  </Link>
                </li>
              </>
            ) : (
              <>
                <li>
                  <Link to={"/profile"} className="route-link">
                    profile
                  </Link>
                </li>
                <li>
                  <button
                    className="logout-button"
                    onClick={handleLogout}
                    disabled={isLogoutDisabled}
                  >
                    logout
                  </button>
                </li>
              </>
            )}
          </ul>
        </div>
      </div>
    </nav>
  );
}
