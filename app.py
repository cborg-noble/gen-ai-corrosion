import deepchem as dc
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

# def visualize_molecule(smiles):
#     st.write(smiles)
#     mol = Chem.MolFromSmiles(smiles)
#     #img = Draw.MolToImage(mol)
#     #st.image(img, caption=smiles, use_column_width=False)
#     d1 = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(300,200)
#     d1.DrawMolecule(mol)
#     d1.FinishDrawing()
#     svg1 = d1.GetDrawingText().replace('svg:','')
#     st.image(svg1)

def visualize_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol)
        st.image(img, caption=smiles, use_column_width=False)
    else:
        pass

def replace_element(starting_smiles, element_to_replace):

    mol = Chem.MolFromSmiles(starting_smiles)

    replacements = ['O', 'C', 'F', 'S', 'N']
    replacements = [x for x in replacements if x != element_to_replace]

    all_new_smiles = []
    for ele in replacements:
        replacement = Chem.MolFromSmiles(ele)
        modified_mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles(element_to_replace), replacement)
        new_smiles = [Chem.MolToSmiles(mol) for mol in modified_mol]
        all_new_smiles.extend(new_smiles)
    
    valid_smiles = [x for x in all_new_smiles if Chem.MolFromSmiles(x)!=None]
    return valid_smiles



def main():
    st.title('Utilizing Chemical Replacements to Generate Novel Corrosion Inhibitor Molecules')

    with st.expander("Background on SMILES and this dataset: "):
        st.write("The simplified molecular-input line-entry system (SMILES) is a line notation \
                 for describing the structure of a molecule in a way that can be used for AI.")
        st.write("Below, I have created a dataset of molecules with high-performing corrosion inhibition properties and have\
                 calculated SMILES strings for each of them.")

    df = pd.read_csv('ci_smiles_ml.csv')
    st.dataframe(df)


    user_smiles = st.text_input('Input a SMILES string and press ENTER: ')

    if user_smiles:
        

        progress_text = "Computing and validating new molecular structures..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()


        all_new_smiles = []
        for ele in ['O', 'C', 'F', 'S', 'N']:
            all_new_smiles.extend(replace_element(user_smiles, ele))
        all_new_smiles = np.unique(all_new_smiles)


        col1, col2 = st.columns(2)

        with col1:
            visualize_molecule(user_smiles)
            st.write('Starting molecule')
        with col2:
            st.write(all_new_smiles)
            st.write('Molecules generated from substructure replacements')


        for index, sm in enumerate(all_new_smiles):
            if index % 2 != 0:
                with col1:
                        visualize_molecule(sm)

            else:
                with col2:
                        visualize_molecule(sm)  


        st.divider()

        st.subheader('How can we identify if any of these have good corrosion resistance?')

        st.markdown(
            """
            We will perform the following:
            1. Train an AI model on our past experiments to predict Corrosion Rate.
            2. Use that model to make predictions on the molecules we just generated above.
            3. Sort them by their predicted corrosion rate and compare to our previous best.
            """)
        
        
        if st.button("Train AI model and generate preditions"):
            df = pd.read_csv('ci_smiles_ml.csv')
            smiles = [str(x).upper() for x in df['SMILES']]

            rdkit_featurizer = dc.feat.RDKitDescriptors()
            features = rdkit_featurizer(smiles)

            df_featurized = pd.DataFrame(features, columns=rdkit_featurizer.descriptors)
            df_featurized['SMILES'] = smiles
            df_featurized['CorrosionRate'] = df['CorrosionRate']

            X = df_featurized[rdkit_featurizer.descriptors].values

            output= 'CorrosionRate'
            y = df_featurized[output]

            rf = RandomForestRegressor()
            rf.fit(X, y)
            y_pred = cross_val_predict(rf, X, y, cv=5)
            df_featurized[output+'_pred'] = y_pred
            rmse = round(mean_squared_error(y, y_pred, squared=False), 3)

            fig = px.scatter(df_featurized, x=output, y=output+'_pred')
            fig.add_trace(go.Scatter(x=[0,1.5], y=[0,1.5],mode='lines'))
            fig.update_traces(marker={'size': 12, 'line_width':2, 'line_color':'DarkSlateGrey'})
            fig.update_layout(height=400, width=400, showlegend=False, font_family='inter', font_size=14)

            fig.update_xaxes(title_font_family="Inter")

            fig.add_annotation(x=0.5, y=1.5,
                        text="RMSE: {}".format(rmse),
                        showarrow=False,
                        arrowhead=1)

            st.plotly_chart(fig, use_container_width=False)


            X_new = rdkit_featurizer(all_new_smiles)
            y_pred_new = rf.predict(X_new)
            data = {'SMILES':all_new_smiles, 'Predicted Corrosion Rate (mpy)':y_pred_new}
            df_new = pd.DataFrame(data).sort_values(by='Predicted Corrosion Rate (mpy)', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.write(df_new)
            with col2:
                visualize_molecule(str(df_new['SMILES'][0]))


if __name__ == '__main__':
    main()

# fig = px.scatter(df, x="AvgProtectPercent", y="CorrosionRate")
# st.plotly_chart(fig, use_container_width=True)
                        
