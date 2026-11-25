// Reads one JSON-encoded string per stdin line (a generated code fragment),
// wraps it in a C# method body, parses it with Roslyn, and writes one JSON
// object per line with the number of syntax errors. Pure syntax diagnostics:
// no compilation, no semantic binding, no network, no judge.

using System;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

internal static class Program
{
    private const string WrapperOpen = "class __W { void __M() {\n";
    private const string WrapperClose = "\n} }";

    private static int Main()
    {
        string line;
        while ((line = Console.ReadLine()) != null)
        {
            string fragment;
            try
            {
                fragment = System.Text.Json.JsonSerializer.Deserialize<string>(line) ?? string.Empty;
            }
            catch
            {
                Console.WriteLine("{\"errors\":-1}");
                continue;
            }
            SyntaxTree tree = CSharpSyntaxTree.ParseText(WrapperOpen + fragment + WrapperClose);
            int errors = tree.GetDiagnostics()
                             .Count(d => d.Severity == DiagnosticSeverity.Error);
            Console.WriteLine("{\"errors\":" + errors + "}");
        }
        return 0;
    }
}
