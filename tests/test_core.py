import unittest
from engine.vars import VarManager
from engine.compilation.artifact import CompilationArtifact

class TestCore(unittest.TestCase):
    def test_var_manager_deterministic(self):
        vm1 = VarManager()
        id1_x = vm1.declare("x")
        id1_y = vm1.declare("y")
        
        vm2 = VarManager()
        id2_x = vm2.declare("x")
        id2_y = vm2.declare("y")
        
        self.assertEqual(id1_x, id2_x)
        self.assertEqual(id1_y, id2_y)
        self.assertNotEqual(id1_x, id1_y)

    def test_var_manager_fresh(self):
        vm = VarManager()
        v1 = vm.fresh("aux")
        v2 = vm.fresh("aux")
        self.assertNotEqual(v1, v2)
        # VarManager uses ::namespace::prefix_id format
        self.assertIn("::default::aux_1", vm.get_var_map())
        self.assertIn("::default::aux_2", vm.get_var_map())

    def test_var_manager_reserve(self):
        vm = VarManager()
        block = vm.reserve_block(5, "block")
        self.assertEqual(len(block), 5)
        self.assertEqual(len(set(block)), 5)
        for i in range(1, 6):
            self.assertIn(f"::default::block_{i}", vm.get_var_map())

    def test_artifact_serialization(self):
        art = CompilationArtifact(
            backend_name="test",
            encoding_config={},
            clauses=[[1, -2]],
            var_map={"x": 1, "y": 2},
            id_to_name={1: "x", 2: "y"},
            constraint_ids=["c1"],
            constraint_to_clause_ids={"c1": [0]},
            aux_vars=set()
        )
        # Should be a dataclass
        self.assertEqual(art.backend_name, "test")

if __name__ == "__main__":
    unittest.main()
