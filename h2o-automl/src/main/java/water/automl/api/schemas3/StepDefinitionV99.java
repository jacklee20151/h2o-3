package water.automl.api.schemas3;

import ai.h2o.automl.StepDefinition;
import ai.h2o.automl.StepDefinition.Alias;
import ai.h2o.automl.StepDefinition.Step;
import water.api.API;
import water.api.EnumValuesProvider;
import water.api.Schema;

public final class StepDefinitionV99 extends Schema<StepDefinition, StepDefinitionV99> {

  public static final class StepV99 extends Schema<Step, StepV99> {
    @API(help="The id of the step (must be unique per step provider).", direction=API.Direction.INOUT)
    public String id;
    @API(help="The relative weight for the given step (can impact time and/or number of models allocated for this step).", direction=API.Direction.INOUT)
    public int weight;
  }  
  
  public static final class AliasProvider extends EnumValuesProvider<Alias> {
    public AliasProvider() {
      super(Alias.class);
    }
  }

  @API(help="Name of the step provider (usually, this is also the name of an algorithm).", direction=API.Direction.INOUT)
  public String name;
  @API(help="An alias representing a predefined list of steps to be executed.", valuesProvider=AliasProvider.class, direction=API.Direction.INOUT)
  public Alias alias;
  @API(help="The list of steps to be executed (Mutually exclusive with alias).", direction=API.Direction.INOUT)
  public StepV99[] steps;

}

