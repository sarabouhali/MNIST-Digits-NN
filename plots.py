import numpy as np
import plotly.graph_objects as go


class DataViz:

    def plotHist(self, activations, unit):
        u_act=[i[unit] for i in activations]
        print (u_act)
        fig = go.Figure(data=[go.Histogram(x=u_act)])
        fig.update_layout(
            title=go.layout.Title(
                text="Activations histogram",
                xref="paper",
            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text="Activations",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text="Count",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            )
        )
        fig.show()
        return

    def plotLoss(self, loss, val_loss):
        fig = go.Figure()
        fig.add_trace(go.Scatter( y=loss,
                                 mode='lines',
                                 name='Training loss'))
        fig.add_trace(go.Scatter( y=val_loss,
                                 mode='lines+markers',
                                 name='Validation Loss'))
        fig.update_layout(
            title=go.layout.Title(
                text="Model loss",
                xref="paper",
            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text="Iterations",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text="Loss",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            )
        )
        fig.show()
        return

    def plotAcc(self, accuracy, v_acc):

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=accuracy,
                                 mode='lines',
                                 name='Training'))
        fig.add_trace(go.Scatter(y=v_acc,
                                 mode='lines+markers',
                                 name='Validation'))
        fig.update_layout(
            title=go.layout.Title(
                text="Model accuracy",
                xref="paper",
            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text="Iterations",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text="Accuracy",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            )
        )
        fig.show()
        return

    def plotWeights(self, w, node):
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        def angles_w(weights, n):
            ang=[]
            for i in range(len(weights)-1):
                ang.append(angle_between(weights[i][n], weights[i+1][n]))
            return ang

        a = angles_w(w, node)
        x = [i for i in range(1, len(a)+1)]

        fig = go.Figure(data=go.Scatter(x=x, y=a))
        fig.update_layout(
            title=go.layout.Title(
                text="Weights evolution through iterations",
                xref="paper",
            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text="Iterations",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text="Angle",
                    font=dict(
                        family="Courier New, monospace",
                        size=18,
                        color="#7f7f7f"
                    )
                )
            )
        )
        fig.show()

        return