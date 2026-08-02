import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import AppShell from "./components/AppShell";
import Login from "./pages/Login";
import Chat from "./pages/Chat";
import Documents from "./pages/Documents";
import AdminUsers from "./pages/AdminUsers";
import AdminObservability from "./pages/AdminObservability";

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/chat"
        element={
          <ProtectedRoute>
            <AppShell>
              <Chat />
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/documents"
        element={
          <ProtectedRoute>
            <AppShell>
              <Documents />
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/users"
        element={
          <ProtectedRoute role="admin">
            <AppShell>
              <AdminUsers />
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/observability"
        element={
          <ProtectedRoute role="admin">
            <AppShell>
              <AdminObservability />
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route path="/" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
