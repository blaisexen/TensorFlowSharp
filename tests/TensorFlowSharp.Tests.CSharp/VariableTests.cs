using System;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class VariableTests
	{
		[Fact(Skip = "Needs #170")]
		public void MultipleVariables_InitializedSeparately_HaveSeparateValues ()
		{
			using (var graph = new TFGraph ()) {
				TFOperation initW;
				TFOutput valueW;
				var W = graph.Variable (graph.Const (1.0), out initW, out valueW, operName: "W");

				TFOperation initb;
				TFOutput valueb;
				var b = graph.Variable (graph.Const (-0.3), out initb, out valueb, operName: "b");

				var y = graph.Add(valueW, valueb);

				using (var sess = new TFSession (graph)) {

					Assert.Throws<TFException> (() => sess.GetRunner ().Fetch (y).Run ()); // ok

					var r1 = sess.GetRunner ().AddTarget (initW).Run ();
					var r2 = sess.GetRunner ().AddTarget (initb).Run ();
					var r = sess.GetRunner ().Fetch (y).Run ();

					Assert.Equal (0.7, (double)r[0].GetValue ()); // fail, will return -0.6 because both vars will be initialized with 0.3
				}
			}
		}

		[Fact]
		public void MultipleVariables_InitializedSeparately_HaveSeparateValues_V2 ()
		{
			using (var graph = new TFGraph ()) {

				var W = graph.VariableV2 (TFShape.Scalar, TFDataType.Double, operName: "W");
				var initW = graph.Assign (W, graph.Const (1.0));

				var b = graph.VariableV2 (TFShape.Scalar, TFDataType.Double, operName: "b");
				var initb = graph.Assign (b, graph.Const (-0.3));

				var y = graph.Add (W, b);

				using (var sess = new TFSession (graph)) {

					Assert.Throws<TFException> (() => sess.GetRunner ().Fetch (y).Run ()); // ok

					var r1 = sess.GetRunner ().AddTarget (initW.Operation).Run ();
					var r2 = sess.GetRunner ().AddTarget (initb.Operation).Run ();
					var r = sess.GetRunner ().Fetch (y).Run ();

					Assert.Equal (0.7, (double)r [0].GetValue ()); // ok
				}
			}
		}

		[Fact]
		public void Should_UpdateAfterRun ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				float aValue = 1;
				float bValue = 2;

				TFOutput a = graph.Placeholder (TFDataType.Float);
				Variable b = graph.Variable (graph.Const (0f), trainable: false);
				TFOutput c = graph.Placeholder (TFDataType.Float);

				var r1 = graph.Add (a, b.Read);
				var r2 = graph.Mul (r1, c);
				var newb = graph.AssignAddVariableOp (b, graph.Const (1f));

				var res = session.Run (new [] { a, b }, new TFTensor [] { aValue, bValue },
					new TFOutput [] { r1, r2 }, new [] { r1.Operation, newb }); // 1+2=3
				var calculated = (float)res [0].GetValue ();
				var updated = (int)res [1].GetValue ();
				Assert.Equal (3, calculated);
				Assert.Equal (1, updated);
			}
		}

		[Fact]
		public void ShouldNot_ChangeTypes()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				float aValue = 1;
				float bValue = 2;

				TFOutput a = graph.Placeholder (TFDataType.Float);
				Variable b = graph.Variable (graph.Const (0), trainable: false);
				TFOutput c = graph.Placeholder (TFDataType.Float);

				var r1 = graph.Add (a, graph.Cast(b.Read, TFDataType.Float));
				var r2 = graph.Mul (r1, c);
				var newb = graph.AssignAddVariableOp (b, graph.Const (1));

				var res = session.Run (new [] { a, b }, new TFTensor [] { aValue, bValue }, 
					new TFOutput [] { r1 }, new [] { r1.Operation, newb }); // 1+2=3
				var calculated = (float)res [0].GetValue ();
				var updated = (int)res[1].GetValue ();
				Assert.Equal (3, calculated);
				Assert.Equal (1, updated);
			}
		}
	}
}
